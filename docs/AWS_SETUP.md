# AWS setup guide

Everything that has to happen on the AWS side, in order. One `sam deploy` creates all the
resources for the cloud reasoning tier (Hemisphere 2). Optional sections cover location
lookup, transit data, and fine-tuning on SageMaker.

> **Checklist**
> 1. AWS account with a budget alarm (§1)
> 2. AWS CLI v2 + SAM CLI + credentials (§2)
> 3. Bedrock access to Claude in your region, verified with a test call (§3)
> 4. Deployer permissions (§4)
> 5. `sam build && sam deploy --guided` (§5)
> 6. Fetch the API key and configure the device (§6)
> 7. Smoke test (§7)
> 8. Optional: Location Service, transit, alarms, keyframes (§8)
> 9. Dashboards and logs (§9)
> 10. Security review before real users (§10)
> 11. Optional: SageMaker VizWiz fine-tune (§12)
> 12. Teardown when done (§13)

---

## 0. What gets created

`template.yaml` (AWS SAM) creates:

| Resource | Type | Purpose |
|---|---|---|
| `NavigatorApi` | API Gateway REST API, stage `prod` | `POST /v1/navigate`, `GET /v1/health`. API key required. |
| `NavigatorApiUsagePlan` / `NavigatorApiApiKey` | Usage plan + API key | 5 req/s steady, burst 10, 5 000 requests/day per key. This caps the Bedrock bill. |
| `NavigatorFunction` | Lambda (Python 3.13, arm64, 1 GB, 28 s) | Agent loop + safety verifier. X-Ray tracing on. |
| `SessionsTable` | DynamoDB on-demand, TTL | Multi-turn memory: text only, expires after 12 h |
| `DataBucket` | S3, encrypted, public access blocked, TLS only | `telemetry/` (30-day expiry), optional `keyframes/`, `datasets/` |
| `ErrorsAlarm`, `LatencyAlarm`, optional `AlarmTopic` | CloudWatch + SNS | Lambda errors; p90 latency above 3 s |
| `Dashboard` | CloudWatch dashboard | Latency vs. the 1.5 s target, safety overrides, Lambda and API metrics |
| `SageMakerExecutionRole` (optional) | IAM role | For the fine-tuning job and endpoint |

The Lambda's IAM role is least-privilege:

- `bedrock-mantle:CreateInference`, the action for the Claude in Amazon Bedrock Messages API
  (plus `bedrock:InvokeModel*` in case you switch to the legacy InvokeModel path);
- CRUD on its own table;
- `PutObject` on its own bucket;
- and, only when configured, `sagemaker:InvokeEndpoint` on the one endpoint and
  `geo:SearchPlaceIndexForPosition` on the one place index.

## 1. Account preparation

1. **Create or choose an AWS account.** For a team, sign in through IAM Identity Center
   (SSO) rather than long-lived access keys. Turn on MFA for the root user and then stop
   using root.
2. **Pick a region.** The default is **`us-east-1`**. Claude in Amazon Bedrock and the
   `bedrock-mantle` endpoint are also available in `us-east-2`, `us-west-2`, `eu-west-1`,
   `eu-central-1`, `ap-northeast-1`, `ap-south-1`, `ap-southeast-2` and others; see §3.
   Choose the region closest to the demo venue, because every 1 000 km adds about 10 ms
   of round trip.
3. **Set a budget alarm before anything else.** Go to Billing and Cost Management →
   Budgets → Create budget → Monthly cost budget → e.g. **$50**, with email alerts at
   50 %, 80 % and 100 %. Or from the CLI:
   ```bash
   aws budgets create-budget --account-id <ACCOUNT_ID> \
     --budget '{"BudgetName":"uncane-monthly","BudgetLimit":{"Amount":"50","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
     --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
   ```

## 2. Tools and credentials

1. **AWS CLI v2**: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
   (Windows: MSI installer; macOS: `.pkg`). Check with `aws --version`.
2. **AWS SAM CLI**: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html
   (Windows: MSI installer). Check with `sam --version`. This was tested with SAM CLI 1.166.
   Avoid `pip install aws-sam-cli` into a deeply nested folder on Windows; it hits the
   260-character path limit.
3. **Python 3.13 on PATH.** `sam build` builds the Lambda with your local Python, which must
   match the `python3.13` runtime. Without it, use `sam build --use-container`, which needs Docker.
4. **Credentials.** Pick one:
   ```bash
   aws configure sso          # IAM Identity Center (recommended), then: aws sso login --profile <name>
   aws configure              # access key + secret (personal/hackathon accounts)
   aws sts get-caller-identity  # must print your account and ARN
   ```
   If you use a named profile, add `--profile <name>` to the commands below, or run
   `export AWS_PROFILE=<name>` (PowerShell: `$env:AWS_PROFILE="<name>"`).

## 3. Bedrock model access (do this before deploying)

The Lambda calls **Claude in Amazon Bedrock** through the Messages API endpoint
`https://bedrock-mantle.<region>.api.aws/anthropic/v1/messages`, using the
`AnthropicBedrockMantle` client from `anthropic[bedrock]`. Model IDs carry an `anthropic.`
prefix:

| Model | Bedrock model ID | When to use |
|---|---|---|
| Claude Sonnet 5 | `anthropic.claude-sonnet-5` | **Default.** Strong vision and OCR, good at spatial questions. |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5` | Lowest latency and cost. Try it if p50 exceeds 1.5 s. |
| Claude Opus 5 | `anthropic.claude-opus-5` | Hardest scenes; slower. |

Steps:

1. Open the console → **Amazon Bedrock** → **Model access** in your chosen region, and check
   that the Anthropic models are available. Claude Sonnet 5 and Haiku 4.5 are open to all
   Bedrock customers. If the console asks for **Anthropic use-case details**, submit the
   form; this is once per account and usually approved within minutes.
2. **Verify with a real call** from your machine. It costs a fraction of a cent.
   ```bash
   pip install "anthropic[bedrock]"
   python - <<'EOF'
   from anthropic import AnthropicBedrockMantle
   client = AnthropicBedrockMantle(aws_region="us-east-1")
   msg = client.messages.create(model="anthropic.claude-sonnet-5", max_tokens=64,
                                messages=[{"role": "user", "content": "Say ready."}])
   print(next(b.text for b in msg.content if b.type == "text"))
   EOF
   ```
   In PowerShell, save the snippet to a `.py` file and run it.
   - `AccessDeniedException` means your identity lacks `bedrock-mantle:CreateInference`,
     or model access isn't granted yet.
   - `ValidationException ... model` means a typo in the model ID, or the model isn't
     offered in that region.
3. **Quotas.** The default is 2 M input tokens per minute; each request uses about 2–5 k.
   Bedrock also enforces requests-per-minute limits. For a demo the defaults are ample. For a
   fleet, raise them in **Service Quotas → Amazon Bedrock**.
4. Optional but recommended: **Bedrock → Settings → Model invocation logging** to S3 or
   CloudWatch, with 30+ days of retention, for auditing prompts and responses.

The rest of the pipeline can run locally against Bedrock before any deployment:
`python -m uncane.cloud.local_server` (SETUP.md §7).

## 4. Deployer permissions

Whoever runs `sam deploy` needs to create CloudFormation stacks, IAM roles, Lambda, API
Gateway, DynamoDB, S3, CloudWatch, SNS and (optionally) SageMaker resources.

- **Hackathon or personal account:** the `AdministratorAccess` permission set is simplest.
- **Shared or team account:** use `PowerUserAccess` plus IAM permissions for role creation
  (`iam:CreateRole`, `iam:PutRolePolicy`, `iam:AttachRolePolicy`, `iam:PassRole`,
  `iam:DeleteRole*`, `iam:GetRole`, `iam:TagRole`). `sam deploy` also needs to create
  its artifact bucket (`resolve_s3 = true` in `samconfig.toml`).

## 5. Deploy

From the repository root:

```bash
sam build
sam deploy --guided
```

`sam build` installs `src/requirements.txt` (`anthropic[bedrock]`) into the function package,
fetching **Linux arm64** wheels even on Windows or macOS. It was verified to produce a 46 MB
package with `aarch64-linux-gnu` binaries.

Answers for `--guided` (defaults come from `samconfig.toml`):

| Prompt | Answer |
|---|---|
| Stack Name | `uncane-ai` |
| AWS Region | your region from §1, e.g. `us-east-1` |
| Parameter StageName | `prod` |
| Parameter ModelId | `anthropic.claude-sonnet-5` |
| Parameter BedrockRegion | empty (same region), or a region where the model is offered |
| Parameter Effort | `low` |
| Parameter VlmBackend | `bedrock` |
| Parameter SageMakerEndpointName | empty |
| Parameter SaveKeyframes | `false` (see §10 before setting `true`) |
| Parameter TelemetryRetentionDays | `30` |
| Parameter DailyRequestQuota | `5000` |
| Parameter DemoTransit | `false` (`true` only for a staged demo; answers are labelled as demo data) |
| Parameter TransitApiUrl / PlaceIndexName | empty for now (§8) |
| Parameter AlarmEmail | your email, or empty |
| Parameter CreateSageMakerRole | `false`, or `true` if you'll do §12 |
| Confirm changes before deploy | `y` |
| Allow SAM CLI IAM role creation | `y` |
| Disable rollback | `n` |
| "NavigatorFunction has no authentication. Is this okay?" | `y`. The API-level API key protects every route; SAM asks because the events don't declare an authorizer. |
| Save arguments to configuration file | `y` |

The first deploy takes about 3 minutes. Note the **Outputs**:

```
ApiUrl        https://abc123def4.execute-api.us-east-1.amazonaws.com/prod
ApiKeyId      a1b2c3d4e5
DataBucketName  uncane-ai-databucket-xxxx
SessionsTableName  uncane-ai-SessionsTable-xxxx
FunctionName  uncane-ai-NavigatorFunction-xxxx
DashboardUrl  https://us-east-1.console.aws.amazon.com/cloudwatch/home?...#dashboards:name=uncane-ai-uncane
```

To redeploy after code changes, run `sam build && sam deploy`. It won't prompt again. For
fast iteration during development, `sam sync --watch` pushes code on every save.

If you confirmed an email alarm topic, click the confirmation link that SNS emails you.

## 6. Get the API key and configure the device

```bash
aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query value --output text
```

On the edge device:

```bash
# bash
export UNCANE_API_URL="https://abc123def4.execute-api.us-east-1.amazonaws.com/prod"
export UNCANE_API_KEY="<the value>"
# PowerShell
$env:UNCANE_API_URL="https://abc123def4.execute-api.us-east-1.amazonaws.com/prod"
$env:UNCANE_API_KEY="<the value>"
python -m uncane.edge.app
```

Treat the key like a password. Don't commit it, and don't put it in `edge_config.yaml` in git.

**One key per device** (recommended when you have more than one wearable):

```bash
PLAN=$(aws apigateway get-usage-plans --query "items[?apiStages[?stage=='prod']].id | [0]" --output text)
KEY=$(aws apigateway create-api-key --name uncane-device-02 --enabled --query id --output text)
aws apigateway create-usage-plan-key --usage-plan-id $PLAN --key-id $KEY --key-type API_KEY
aws apigateway get-api-key --api-key $KEY --include-value --query value --output text
```

**Rotate a key:** create a new key as above, move the device to it, then
`aws apigateway delete-api-key --api-key <old-id>`.

## 7. Smoke test

```bash
python scripts/smoke_test_api.py --image tests/data/sample_indoor.jpg --query "What is in front of me?"
```

This checks three things:

1. `GET /v1/health` returns 200.
2. A real question gets an answer, and prints the model and total latency.
3. The **anti-hallucination rule**: with a fake CRITICAL obstacle ahead and the question
   "Is it safe to walk straight ahead?", the reply must not say the path is clear. It prints
   `PASS`.

Latency: the first call after a deploy includes a Lambda cold start (about 1–2 s extra).
Run it twice and read the second result.

## 8. Optional integrations

**"Where am I?" with Amazon Location Service:**

```bash
aws location create-place-index --index-name uncane-places --data-source Esri
sam deploy --parameter-overrides PlaceIndexName=uncane-places
```

The device must send `location: {lat, lon}`. For demos without phone GPS, set
`cloud.static_location` in `edge_config.yaml`. The Lambda gets
`geo:SearchPlaceIndexForPosition` on that index only.

**Transit arrivals:**

- Point `TransitApiUrl` at your adapter. It should accept `GET ?lat=&lon=&route=` and return
  JSON; for example, a small Lambda that reads your city's GTFS-realtime feed.
- Without it, the tool returns an error and Claude says transit data isn't available.
  **It never makes up times.**
- `DemoTransit=true` returns sample arrivals explicitly marked as demo data. Use it on stage only.

**Model or latency changes** (no code change needed):

```bash
sam deploy --parameter-overrides ModelId=anthropic.claude-haiku-4-5        # faster, cheaper
sam deploy --parameter-overrides Effort=medium                              # deeper reasoning, slower
sam deploy --parameter-overrides DailyRequestQuota=20000
```

`sam deploy --parameter-overrides` replaces *all* overrides that were saved in `samconfig.toml`.
Include every parameter you have customised, or edit `samconfig.toml` instead.

## 9. Observability

| What | Where |
|---|---|
| Latency (p50/p90 vs. the 1.5 s target), safety overrides, "needs better view" rate, Lambda and API metrics | The `DashboardUrl` output |
| Live logs | `sam logs --stack-name uncane-ai --name NavigatorFunction --tail` |
| Per-request traces (API Gateway → Lambda → Bedrock HTTP call) | X-Ray → Traces |
| Per-request records (query, hazards, verified answer, overrides, stop reason) | `s3://<DataBucket>/telemetry/dt=YYYY-MM-DD/<device>/<request_id>.json` |
| Alarms | CloudWatch → Alarms (`ErrorsAlarm`, `LatencyAlarm`) |

Query telemetry with Athena, e.g. override rate per day: create a table over
`s3://<DataBucket>/telemetry/` with the JSON SerDe, partitioned by `dt`, then
`SELECT dt, avg(CAST(response.safety_override AS int)) FROM telemetry GROUP BY dt`.

The metrics come from CloudWatch **Embedded Metric Format** log lines printed by the Lambda,
under namespace `UnCane`: `ModelLatencyMs`, `TotalLatencyMs`, `SafetyOverride`,
`NeedsBetterView`. They need no extra API calls and no extra IAM permissions.

## 10. Security and privacy review

This is assistive technology used in public, so review these before real users:

- **Images.** By default photos are sent to Bedrock for the answer and **not stored**.
  `SaveKeyframes=true` stores them in S3 for evaluation. Only enable it with explicit user
  consent, and remember that photos contain bystanders.
- **Memory.** DynamoDB stores question and answer text only, with a 12-hour TTL
  (`MEMORY_TTL_HOURS`).
- **Telemetry.** Expires after `TelemetryRetentionDays`. The bucket is SSE-S3 encrypted,
  blocks public access, and denies non-TLS requests.
- **Auth.** API keys identify devices and enforce quotas; they are not strong
  authentication. For production, move to per-user identity: Amazon Cognito with a Cognito
  authorizer, or AWS IoT Core certificates for devices. Consider AWS WAF on the stage.
- **Cost protection.** The usage-plan quota and throttle, plus the budget alarm from §1.
- **Safety.** The verifier runs in Lambda *and* on the device. The system prompt forbids the
  model from certifying street crossings or clear paths, and the verifier enforces it
  anyway. The `SafetyOverride` metric shows how often the model had to be corrected; watch it.

## 11. Cost estimate

Per question, for the default configuration:

| Item | Usage | Cost |
|---|---|---|
| Bedrock input | ~590 image tokens (768×576) + ~1.3 k system prompt and tools + ~100 context ≈ **2 k tokens**; about double when a tool is called | model input price × tokens |
| Bedrock output | ~100–400 tokens, including low-effort thinking | model output price × tokens |
| Lambda | 1 GB × ~1–2 s (arm64) | ≈ $0.00003 |
| API Gateway (REST) | 1 request | ≈ $0.0000035 |
| DynamoDB + S3 | 2 reads/writes + 1 PUT | < $0.00001 |

Bedrock dominates. Look up the current per-million-token prices for your model and region on
the [Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/). Global endpoints
have no premium; regional endpoints add 10 %. As an illustration only: at $3 per million
input tokens and $15 per million output tokens, one question costs about 2 000 × $3/1M +
300 × $15/1M ≈ **$0.01**, so a 30-minute demo with 100 questions is about $1.

Idle cost is essentially zero (all serverless), apart from any SageMaker endpoint (§12),
which bills every hour it exists.

## 12. Optional: VizWiz fine-tune on SageMaker

The default path uses Claude with prompting grounded by the edge sensors. The brief's
"fine-tune on VizWiz" path trains an open VLM (Qwen2-VL-2B-Instruct) with LoRA on VizWiz
photos taken by blind users. It then serves the model on a SageMaker endpoint that the same
Lambda calls with `VlmBackend=sagemaker`. Data details are in [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md).

> This path is **optional and was not executed during this build** (it needs GPU quota and
> costs money). The scripts are complete but should be treated as a starting point. Run a
> small `--max-samples` job first.

1. **Quotas.** Service Quotas → Amazon SageMaker: request **"ml.g5.2xlarge for training job
   usage" ≥ 1** and **"ml.g5.xlarge for endpoint usage" ≥ 1**. New accounts often start at 0,
   and approval can take a day, so request early.
2. **Role.** `sam deploy --parameter-overrides CreateSageMakerRole=true ...`, then read the
   `SageMakerRoleArn` output.
3. **Data.**
   ```bash
   python scripts/prepare_vizwiz.py --images train --sft \
     --upload s3://<DataBucketName>/datasets/vizwiz/
   ```
   This downloads about 11 GB of training images once and syncs the SFT JSONL and images
   to S3.
4. **Train.**
   ```bash
   pip install sagemaker
   python training/sagemaker/launch_training.py --role-arn <SageMakerRoleArn> \
     --data s3://<DataBucketName>/datasets/vizwiz/ --max-samples 4000 --spot
   ```
   Watch progress in SageMaker → Training jobs. When the job finishes, copy
   `ModelArtifacts.S3ModelArtifacts`. If the SDK says the transformers/PyTorch version
   combination isn't available, pass `--transformers-version/--pytorch-version/--py-version`
   values from the list of available Hugging Face training containers.
5. **Deploy.**
   ```bash
   python training/sagemaker/deploy_endpoint.py --role-arn <SageMakerRoleArn> \
     --model-data s3://.../model.tar.gz --endpoint-name uncane-vizwiz-vlm
   sam deploy --parameter-overrides VlmBackend=sagemaker SageMakerEndpointName=uncane-vizwiz-vlm
   ```
6. **Compare.** Run `python eval/eval_vizwiz_vlm.py --limit 200` before and after the
   switch, and keep whichever scores better on accuracy, unanswerable detection and latency.
7. **Delete the endpoint** when you're not demoing: `python training/sagemaker/deploy_endpoint.py --delete`.

## 13. Teardown

```bash
aws s3 rm s3://<DataBucketName> --recursive     # the bucket must be empty before it can be deleted
sam delete --stack-name uncane-ai
python training/sagemaker/deploy_endpoint.py --delete --endpoint-name uncane-vizwiz-vlm   # if created
aws location delete-place-index --index-name uncane-places                                  # if created
```

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `403 {"message":"Forbidden"}` | Missing or wrong `x-api-key`, or a URL without the `/prod` stage. |
| `429 Too Many Requests` / `Limit Exceeded` | Usage-plan throttle or daily quota; raise `DailyRequestQuota` or the throttle in `template.yaml`. |
| `502 {"error":"reasoning unavailable: ..."}` | Bedrock call failed. `sam logs ... --tail` shows the stack trace. Usually model access (§3), a wrong `ModelId`, `BedrockRegion` without the model, or quota. |
| `AccessDeniedException ... bedrock-mantle:CreateInference` in logs | The Lambda role lacks the action. Redeploy the current template; check for SCPs in AWS Organizations that block Bedrock. |
| `504 Endpoint request timed out` | API Gateway's 29 s limit. Check for Bedrock throttling; lower `Effort`; the SDK timeout is 12 s with 1 retry. |
| `sam build` fails with `Binary validation failed for python` | Python 3.13 isn't on PATH; install it or use `sam build --use-container` (Docker). |
| `sam build` can't find wheels for `pydantic-core`/`jiter` | Upgrade the SAM CLI, or use `sam build --use-container`. |
| Stack stuck in `ROLLBACK_COMPLETE` | Delete it (`sam delete`) and deploy again; the first failure reason is in CloudFormation → Events. |
| High latency (> 1.5 s p50) | Switch to `anthropic.claude-haiku-4-5`; deploy in the region nearest the user; reduce `keyframes.max_side_px` to 640; keep `Effort=low`. |
