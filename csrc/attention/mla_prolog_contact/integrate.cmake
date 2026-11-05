# Build the same bridge in the ordinary vLLM-Ascend extension when enabled.
option(VLLM_ASCEND_ENABLE_CONTACT_SERIAL "Build Contact serial Vector binding" OFF)
if(VLLM_ASCEND_ENABLE_CONTACT_SERIAL)
  target_sources(vllm_ascend_C PRIVATE csrc/attention/mla_prolog_contact/torch_binding.cpp)
endif()
