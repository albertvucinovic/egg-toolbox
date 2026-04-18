curl -sN http://127.0.0.1:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Say hi."}],"max_tokens":40,"stream":false}' \
  | python3 -m json.tool 
