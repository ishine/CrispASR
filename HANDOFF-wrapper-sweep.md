# Handoff: mirror qwen3-tts variant + Orpheus surface across 5 wrappers

## What's done (origin/main as of 1f62647)

C-ABI exports in `src/crispasr_c_api.cpp` for the unified Session API:

```c
int   crispasr_session_set_codec_path(s, path);
int   crispasr_session_set_voice(s, path, ref_text_or_null);
int   crispasr_session_set_speaker_name(s, name);     // orpheus + qwen3-tts CV
int   crispasr_session_n_speakers(s);                 // orpheus + qwen3-tts CV
const char* crispasr_session_get_speaker_name(s, i);  // orpheus + qwen3-tts CV
int   crispasr_session_set_instruct(s, instruct);     // qwen3-tts VoiceDesign
int   crispasr_session_is_custom_voice(s);            // variant detect
int   crispasr_session_is_voice_design(s);            // variant detect
float* crispasr_session_synthesize(s, text, &n);
```

Python `_binding.py` exposes all of the above (verified end-to-end:
1.7B-CustomVoice "vivian" + 1.7B-VoiceDesign instruct both round-trip
via parakeet-v3 word-exact). HF GGUFs at `cstr/qwen3-tts-{1.7b-base,
1.7b-voicedesign}-GGUF` (CustomVoice 1.7B GGUFs sit locally at
`/Volumes/backups/ai/crispasr-models/qwen3-tts-12hz-1.7b-customvoice-{f16,q8_0}.gguf`,
not yet uploaded — separate task).

## What's missing

Rust / Go / Java / Ruby / JS bindings have **`set_speaker_name`**
(landed in commit `4f476c3`) but are missing:

- `set_instruct(handle, instruct) -> int`
- `is_custom_voice(handle) -> bool`
- `is_voice_design(handle) -> bool`

Without these, qwen3-tts VoiceDesign isn't reachable from non-Python
wrappers, and callers can't detect the variant to pick the right
voice-prompt API. Ruby + JS in particular are at parity with the
older `set_speaker_name` rollout but haven't been swept since.

## Files to touch

Mirror commit `4f476c3` ("feat(wrappers): expose Session TTS +
kokoro_resolve_for_lang"):

| Binding | File | Add 3 fns |
|---|---|---|
| Rust | `crispasr-sys/src/lib.rs` | `extern "C"` decls + idiomatic safe wrappers |
| Go | `bindings/go/crispasr_session.go` | C decls + `(s *CrispasrSession) SetInstruct/IsCustomVoice/IsVoiceDesign` |
| Java | `bindings/java/src/main/java/io/github/ggerganov/whispercpp/CrispasrSession.java` | JNA decl in `Lib` interface + public methods |
| Ruby | `bindings/ruby/ext/ruby_crispasr_session.c` | `extern int` + `static VALUE rb_session_*` + `rb_define_module_function` |
| JS (emscripten) | `bindings/javascript/emscripten.cpp` | `extern` decl + `crispasr_session_*` JS-level wrappers |

Header doc-comments + the `set_voice` block already enumerate the
contract per backend. Match that style. JS should also expose a
session-level method (the existing surface uses a `g_tts_session`
global).

## Test once each binding has the fns

Use `/Volumes/backups/ai/crispasr-models/qwen3-tts-12hz-1.7b-voicedesign-q8_0.gguf`
+ `qwen3-tts-tokenizer-12hz.gguf` and exercise the three new methods.
Reference test (Python, all five new methods):

```python
s = Session(model_path=VD_MODEL, backend='qwen3-tts',
            lib_path=DYLIB, n_threads=4)
s.set_codec_path(TOKENIZER)
assert s.is_voice_design() and not s.is_custom_voice()
s.set_instruct("young female with British accent, energetic")
pcm = s.synthesize("Hello from the wrapper.")  # peak ≥ 8000, RMS ≥ 1000
```

Equivalent should land in each binding's existing test file or
example. CustomVoice path uses the 1.7B-CV GGUF (or 0.6B-CV, already
on HF) + `set_speaker_name("vivian")`.

## Why now

Unblocks PLAN #57 Phase 2 cheap registry adds — Kartoffel_Orpheus_DE
+ lex-au are XS-effort once the Orpheus C-ABI surface is reachable
from every wrapper. Same pattern: registry alias + checkpoint swap.

## Suggested commit shape

One squashed commit per binding (5 commits total) following the
`4f476c3` pattern, or one big sweep if the diff stays under ~500
lines. Title example: `feat(wrappers): expose set_instruct +
is_voice_design/is_custom_voice (PLAN #57 Phase 1 follow-up)`.
