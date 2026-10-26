# How to show that the agent calls the OpenCV 5 tools

Two parts: run it **locally**, then run it on **AWS**. Follow them in order. In both parts you are
looking for the same evidence: an `agent_trace` listing each OpenCV tool call, and answers that change
because of what the tool measured.

**What is proven where**

| | Local | AWS |
|---|---|---|
| Tools exist and work | Yes | Only after you redeploy (section 4) |
| Answer depends on tool output | Yes (mock, scripted) or yes (real model, section 3) | Yes, once redeployed |
| Proves "runs on AWS" for the competition | No (the model call goes to Bedrock, but the logic runs on your PC) | Yes |

The deployed stack predates the tools. Until you redeploy it, its replies have no `agent_trace`.

---

## 1. Setup (once per terminal)

```powershell
cd C:\Users\Z0052TRF\Desktop\opencv\uncane
.\.venv\Scripts\Activate.ps1
python -c "import cv2; print(cv2.__version__)"      # must print 5.x
```
Environment variables only exist in the terminal where you set them. Never paste keys into chat or
commit them.

---

## 2. Part A: local, no keys needed

### A1. Offline proof (no server)

```powershell
python eval/eval_agent_tools.py
python -m pytest tests/test_vision_tools.py -q
```
Expected: `6/6 cases passed`, and the traces are saved in `eval/agent_traces.json`. Open that file. The
sharp sign shows `assess_image_quality`, `find_text_regions` and `zoom_region`. The blurry sign shows only
`assess_image_quality`.

### A2. Local server with the scripted mock

Terminal 1 (leave it running):
```powershell
$env:VLM_BACKEND = "mock"
python -m uncane.cloud.local_server
```
Wait for `local cloud tier on http://127.0.0.1:8787`.

Terminal 2:
```powershell
python scripts/show_agent_tools.py
```
Expected output (times vary):
```
--- sharp sign
  answer         : The sign says: Exit, stairs to the left.
  tools_used     : ['assess_image_quality', 'find_text_regions', 'zoom_region', 'final_answer']
  1. [ok ] assess_image_quality ...  opencv=5.0.0
  2. [ok ] find_text_regions ...
  3. [ok ] zoom_region ...           image_back=True
--- blurred sign
  answer         : The photo is too blurry to read. Hold still, aim at the sign and ask again.
  needs re-aim   : True
  tools_used     : ['assess_image_quality', 'final_answer']
RESULT: 4 OpenCV tool call(s) recorded.
```
This is the key moment: **the same question gets a different answer because OpenCV measured a
different photo.** The mock follows a fixed script, so call it "scripted" if asked.

### A3. Local server with the real model (Qwen)

This shows the model choosing tools itself. Stop the mock server, then in terminal 1:
```powershell
$line = Select-String -Path ..\.env -Pattern '^OPENAI_API_KEY=' | Select-Object -First 1
$env:MANTLE_API_KEY = ($line.Line -split '=', 2)[1]
$env:VLM_BACKEND = "mantle"
python -m uncane.cloud.local_server
```
Then in terminal 2 run `python scripts/show_agent_tools.py` again.
- Each request takes several seconds, because every tool round is another model call.
- If a case shows `agent_trace: EMPTY`, the model chose not to call a tool. Run it twice, and if it keeps
  happening, tell me. The prompt wording in `src/uncane/common/prompts.py` needs tuning.
- Real tool use is the number the report should say is "measured separately". Run the script 5 to 10
  times and note how often the tools were called.

### A4. Through the edge app (optional)

Terminal 3, with the local server still running:
```powershell
$env:UNCANE_API_URL = "http://127.0.0.1:8787"; $env:UNCANE_API_KEY = "local"
python -m uncane.edge.app --set cloud.timeout_s=40 --set cloud.max_answer_age_s=40
```
Hold a sign in view and type `read the sign aloud`. After the answer the log shows a line like
`agent OpenCV tools: assess_image_quality (6 ms) -> find_text_regions (2 ms) -> zoom_region (90 ms)`.
The longer limits stop a slow real-model answer being dropped as stale.

---

## 3. What to show for the local part (about one minute)

1. The terminal showing `OpenCV 5.0.0`.
2. `show_agent_tools.py` output: sharp sign (three tools), then blurred sign (one tool, re-aim).
3. Point at `image_back=True` on `zoom_region`: the zoomed image went back to the model.
4. Open `eval/agent_traces.json` or `eval/live_traces.json` as the audit trail.

---

## 4. Part B: AWS

### B1. Redeploy so the tools are in the Lambda

From `uncane/` (needs the AWS CLI and SAM CLI, already installed per your deployment log):
```powershell
sam build
sam deploy --guided
```
- `src/requirements.txt` now includes `opencv-python-headless` and `numpy`, so the build bundles them.
- If `sam build` fails on an OpenCV wheel, use `sam build --use-container` (needs Docker). The Lambda is
  arm64, and I have not tested this build.
- Check the size: the unzipped package must stay under 250 MB.
- `--guided` asks for `MantleApiKey`. Type it there, and do not put it on the command line.
- Keep the other answers the same as the current deployment: `VlmBackend=mantle`, model
  `qwen.qwen3-vl-235b-a22b-instruct`, region `us-east-1`.
- Do this early. If the deploy fails, you still have the local demo, and the old stack keeps running.

### B2. Get the URL and key (without printing the key)

```powershell
aws cloudformation describe-stacks --stack-name uncane-ai --query "Stacks[0].Outputs" --output table
$env:UNCANE_API_KEY = aws apigateway get-api-key --api-key m8tapxez22 --include-value --query value --output text
```
The URL is the `ApiUrl` output (from your log: `https://2m01c57t8l.execute-api.us-east-1.amazonaws.com/prod`).

### B3. Run the same script against AWS

```powershell
python scripts/show_agent_tools.py --url https://2m01c57t8l.execute-api.us-east-1.amazonaws.com/prod --insecure
```
`--insecure` skips TLS checks, only because of your network's certificate problem. Drop it if it works
without. Expect the same two cases as in A2. The first call may be slow while the Lambda starts up.

### B4. Prove it from the AWS side

The Lambda now logs each tool call. Get the function name from the stack output `FunctionName`, then:
```powershell
aws logs tail /aws/lambda/<FunctionName> --since 15m --filter-pattern agent_trace
```
Each line reads `agent_trace request_id=... tools=[...] trace=[...]`.

For the stored record (one JSON file per request):
```powershell
aws s3 ls s3://<DataBucketName>/telemetry/ --recursive | Select-Object -Last 5
aws s3 cp s3://<DataBucketName>/telemetry/<path-from-above>.json -
```
The record includes the full response with `agent_trace`. Photos are not stored unless `SaveKeyframes` is
true. Also open the dashboard from the `DashboardUrl` output to show invocations and latency.

### B5. If the trace is empty on AWS

| Symptom | Likely cause | Fix |
|---|---|---|
| Script says `EMPTY` and the log has no `agent_trace` line | The Lambda still runs old code | Check `sam deploy` finished and used the new build |
| Trace entries show `OpenCV 5 is not available` | The package has no working OpenCV 5 for arm64 | Rebuild with `--use-container`, check the wheel |
| `agent_trace` line exists but is `[]` | The model chose not to call tools | Retry; tune the prompt |
| HTTP 502 or 504 | Several tool rounds exceeded the 28 s Lambda or 29 s API Gateway limit | Retry; reduce image size; try a simpler question |
| HTTP 403 | Wrong API Gateway key | Re-read it in B2 |
| `CERTIFICATE_VERIFY_FAILED` | Your network's proxy | Add `--insecure` |

---

## 5. What to say, and what not to claim

- **Local, scripted:** "The tools, the data flow and the logging are proven, and the answer changes with
  what OpenCV measures."
- **Local or AWS, real model:** report how often it called a tool, from your own runs.
- **Only if the deploy succeeded and B3/B4 show a trace:** "The tools run inside the AWS Lambda on
  Graviton." Otherwise say the tools run locally and the AWS stack provides the model hosting and API.
- Do not say the model "always" uses the tools. Do not hide that the mock is scripted.

---

## 6. Clean-up and safety

- To remove the stack and stop charges: `sam delete`.
- Rotate the AWS access key and the Bedrock key, because they were shown in chat earlier. After rotating,
  the deployed Lambda keeps the old Mantle key until you redeploy.
- `eval/live_traces.json` holds sample answers only, with no photos. It is safe to share.
