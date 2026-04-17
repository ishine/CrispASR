//! Safe Rust wrapper for CrispASR speech recognition.
//!
//! # Quick start
//!
//! ```no_run
//! use crispasr::CrispASR;
//!
//! let model = CrispASR::new("ggml-base.en.bin").unwrap();
//! let segments = model.transcribe_pcm(&pcm_f32).unwrap();
//! for seg in &segments {
//!     println!("[{:.1}s - {:.1}s] {}", seg.start, seg.end, seg.text);
//! }
//! ```

use std::ffi::{CStr, CString};

/// A transcription segment with timing information.
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start: f64, // seconds
    pub end: f64,   // seconds
    pub no_speech_prob: f32,
}

/// A loaded CrispASR model.
///
/// Not `Sync` — do not share between threads.
pub struct CrispASR {
    ctx: *mut crispasr_sys::WhisperContext,
}

unsafe impl Send for CrispASR {}

impl CrispASR {
    /// Load a GGUF/GGML whisper model file.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let path = CString::new(model_path)
            .map_err(|e| format!("invalid path: {e}"))?;
        let cparams = unsafe { crispasr_sys::whisper_context_default_params_by_ref() };
        let ctx = unsafe {
            crispasr_sys::whisper_init_from_file_with_params(path.as_ptr(), cparams)
        };
        unsafe { crispasr_sys::whisper_free_context_params(cparams) };
        if ctx.is_null() {
            return Err(format!("failed to load model: {model_path}"));
        }
        Ok(Self { ctx })
    }

    /// Transcribe raw PCM audio (float32, mono, 16kHz).
    ///
    /// Returns a list of segments with text and timing.
    pub fn transcribe_pcm(&self, pcm: &[f32]) -> Result<Vec<Segment>, String> {
        self.transcribe_pcm_with_strategy(pcm, crispasr_sys::WHISPER_SAMPLING_GREEDY)
    }

    /// Transcribe with a specific sampling strategy.
    pub fn transcribe_pcm_with_strategy(
        &self,
        pcm: &[f32],
        strategy: i32,
    ) -> Result<Vec<Segment>, String> {
        let params = unsafe {
            crispasr_sys::whisper_full_default_params_by_ref(strategy)
        };

        let ret = unsafe {
            crispasr_sys::whisper_full(
                self.ctx,
                params,
                pcm.as_ptr(),
                pcm.len() as i32,
            )
        };
        unsafe { crispasr_sys::whisper_free_params(params) };

        if ret != 0 {
            return Err(format!("transcription failed (error code {ret})"));
        }

        let n = unsafe { crispasr_sys::whisper_full_n_segments(self.ctx) };
        let mut segments = Vec::with_capacity(n as usize);

        for i in 0..n {
            let text_ptr = unsafe {
                crispasr_sys::whisper_full_get_segment_text(self.ctx, i)
            };
            let text = if text_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(text_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            let t0 = unsafe { crispasr_sys::whisper_full_get_segment_t0(self.ctx, i) };
            let t1 = unsafe { crispasr_sys::whisper_full_get_segment_t1(self.ctx, i) };
            let nsp = unsafe {
                crispasr_sys::whisper_full_get_segment_no_speech_prob(self.ctx, i)
            };

            segments.push(Segment {
                text,
                start: t0 as f64 / 100.0,
                end: t1 as f64 / 100.0,
                no_speech_prob: nsp,
            });
        }

        Ok(segments)
    }

    /// Get the detected language from the last transcription.
    pub fn detected_language(&self) -> String {
        let id = unsafe { crispasr_sys::whisper_full_lang_id(self.ctx) };
        let ptr = unsafe { crispasr_sys::whisper_lang_str(id) };
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for CrispASR {
    fn drop(&mut self) {
        unsafe { crispasr_sys::whisper_free(self.ctx) }
    }
}

// =========================================================================
// Unified session — any CrispASR-supported backend through one handle.
//
// Prefer `Session::open` over `CrispASR::new` for new code: it dispatches
// automatically to whichever backend (Whisper, Parakeet, Canary, Cohere,
// Qwen3-ASR, Granite, FastConformer-CTC, Voxtral family, Wav2Vec2) the
// GGUF metadata specifies. `CrispASR` stays around for low-overhead
// Whisper-specific access and ABI stability.
// =========================================================================

/// Word-level timing (populated by backends that produce it).
#[derive(Debug, Clone)]
pub struct SessionWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A segment of a unified-session transcription.
#[derive(Debug, Clone)]
pub struct SessionSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Vec<SessionWord>,
}

/// A loaded session over a CrispASR model of any backend.
pub struct Session {
    handle: *mut crispasr_sys::CrispasrSession,
}

// Not `Sync` — do not share between threads without external sync.
unsafe impl Send for Session {}

impl Session {
    /// Open a GGUF model, auto-detecting the backend from metadata.
    pub fn open(model_path: &str) -> Result<Self, String> {
        Self::open_inner(model_path, None, 4)
    }

    /// Open with an explicit backend (skips auto-detect).
    pub fn open_with_backend(model_path: &str, backend: &str, n_threads: i32) -> Result<Self, String> {
        Self::open_inner(model_path, Some(backend), n_threads)
    }

    fn open_inner(model_path: &str, backend: Option<&str>, n_threads: i32) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let handle = if let Some(be) = backend {
            let be_c = CString::new(be).map_err(|e| format!("invalid backend: {e}"))?;
            unsafe {
                crispasr_sys::crispasr_session_open_explicit(path.as_ptr(), be_c.as_ptr(), n_threads)
            }
        } else {
            unsafe { crispasr_sys::crispasr_session_open(path.as_ptr(), n_threads) }
        };
        if handle.is_null() {
            let avail = Self::available_backends().join(",");
            return Err(format!(
                "Failed to open {model_path:?}. Library was built with: [{avail}]"
            ));
        }
        Ok(Self { handle })
    }

    /// List of backend names the loaded CrispASR library was compiled with.
    pub fn available_backends() -> Vec<String> {
        let mut buf = [0i8; 256];
        let n = unsafe {
            crispasr_sys::crispasr_session_available_backends(buf.as_mut_ptr(), buf.len() as i32)
        };
        if n <= 0 {
            return Vec::new();
        }
        let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
        cstr.to_string_lossy()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Detect the backend from a GGUF file without opening it.
    pub fn detect_backend(model_path: &str) -> Result<String, String> {
        let path = CString::new(model_path).map_err(|e| format!("invalid path: {e}"))?;
        let mut buf = [0i8; 64];
        let n = unsafe {
            crispasr_sys::crispasr_detect_backend_from_gguf(
                path.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        };
        if n <= 0 {
            return Err(format!("backend detection failed (code {n})"));
        }
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Backend name this session ended up using.
    pub fn backend(&self) -> String {
        let p = unsafe { crispasr_sys::crispasr_session_backend(self.handle) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    /// Transcribe 16 kHz mono `f32` PCM. The internal dispatcher routes
    /// to whichever backend this session was opened with.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<SessionSegment>, String> {
        if pcm.is_empty() {
            return Ok(Vec::new());
        }
        let res = unsafe {
            crispasr_sys::crispasr_session_transcribe(self.handle, pcm.as_ptr(), pcm.len() as i32)
        };
        if res.is_null() {
            return Err(format!(
                "crispasr_session_transcribe failed for backend {:?}",
                self.backend()
            ));
        }

        let mut out = Vec::new();
        unsafe {
            let n = crispasr_sys::crispasr_session_result_n_segments(res);
            for i in 0..n {
                let tp = crispasr_sys::crispasr_session_result_segment_text(res, i);
                let text = if tp.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(tp).to_string_lossy().into_owned()
                };
                let t0 = crispasr_sys::crispasr_session_result_segment_t0(res, i) as f64 / 100.0;
                let t1 = crispasr_sys::crispasr_session_result_segment_t1(res, i) as f64 / 100.0;

                let wn = crispasr_sys::crispasr_session_result_n_words(res, i);
                let mut words = Vec::with_capacity(wn as usize);
                for j in 0..wn {
                    let wtp = crispasr_sys::crispasr_session_result_word_text(res, i, j);
                    let wt = if wtp.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(wtp).to_string_lossy().into_owned()
                    };
                    words.push(SessionWord {
                        text: wt,
                        start: crispasr_sys::crispasr_session_result_word_t0(res, i, j) as f64
                            / 100.0,
                        end: crispasr_sys::crispasr_session_result_word_t1(res, i, j) as f64
                            / 100.0,
                    });
                }
                out.push(SessionSegment {
                    text: text.trim().to_string(),
                    start: t0,
                    end: t1,
                    words,
                });
            }
            crispasr_sys::crispasr_session_result_free(res);
        }
        Ok(out)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { crispasr_sys::crispasr_session_close(self.handle) }
    }
}
