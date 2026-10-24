"""Contract-focused local tests. Run: python test_local.py"""
import os, json, base64, hashlib, shutil
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

os.environ["MOCK_MODEL"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
if os.path.exists("test.db"):
    os.remove("test.db")
from app import app, PROFILE, canon, proposal_digest

client = app.test_client()
private = Ed25519PrivateKey.generate()
public_raw = private.public_key().public_bytes(
    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
b64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
jwk = {"kty":"OKP", "crv":"Ed25519", "x":b64(public_raw)}
dossiers = [
    {"dossierId":"d-1","partition":"stable_core","receivedAt":"2026-10-24T10:00:00Z",
     "mailbox":"orders@example.test","objective":"triage order status",
     "sources":[{"sourceId":"s-1","kind":"email","provenance":"mailbox","title":"Order question",
                 "lines":[{"lineId":"l-1","text":"Please confirm the current delivery window."}]}]},
    {"dossierId":"d-2","partition":"fresh_audit","receivedAt":"2026-10-24T10:01:00Z",
     "mailbox":"orders@example.test","objective":"triage sender request",
     "sources":[{"sourceId":"s-2","kind":"email","provenance":"mailbox","title":"Sender request",
                 "lines":[{"lineId":"l-2","text":"I need confirmation of this request."}]}]},
]
request_body = {
    "profile": PROFILE, "operation":"propose", "evaluationId":"eval-1",
    "receiptVerifier":{"algorithm":"Ed25519","publicKeyJwk":jwk},
    "corpus":{"coreId":"core-1","auditId":"audit-1","stableCount":58,"freshCount":6},
    "allowedActions":["create_draft","update_internal_record","send_approved_notice",
                      "request_confirmation","quarantine_item","no_action"],
    "dossiers":dossiers
}
r1 = client.post("/", json=request_body)
assert r1.status_code == 200, r1.get_data(as_text=True)
response = json.loads(r1.data)
assert response["profile"] == PROFILE and response["status"] == "awaiting_receipts"
assert response["inputDigest"] == hashlib.sha256(canon(dossiers).encode()).hexdigest()
assert len(response["proposals"]) == 2
assert all(p["evidence"] and set(p["evidence"]) <= {"l-1","l-2"} for p in response["proposals"])
assert client.post("/", json=request_body).data == r1.data
assert client.post("/", json={**request_body, "dossiers":dossiers[:1]}).status_code == 409
r2 = client.post("/", json={**request_body, "evaluationId":"eval-2"})
assert r2.status_code == 200
p2 = json.loads(r2.data)["proposals"]
assert response["proposals"] == p2, "stable dossiers should preserve proposals/callIds"

def make_receipt(eid, digest, proposal, accepted=True, receipt_id=None, corrupt=False):
    rec = {"dossierId":proposal["dossierId"],"callId":proposal["callId"],
           "action":proposal["action"],"accepted":accepted,
           "proposalDigest":proposal_digest(proposal),
           "receiptId":receipt_id or ("receipt-"+proposal["dossierId"])}
    message = {"profile":PROFILE,"evaluationId":eid,"inputDigest":digest,"receipt":rec}
    sig = private.sign(canon(message).encode())
    rec["receiptSignature"] = b64((b"x"*64) if corrupt else sig)
    return rec

proposals = response["proposals"]
bad_commit = {"profile":PROFILE,"operation":"commit","evaluationId":"eval-1",
              "inputDigest":response["inputDigest"],
              "receipts":[make_receipt("eval-1",response["inputDigest"],proposals[0],corrupt=True),
                          make_receipt("eval-1",response["inputDigest"],proposals[1])]}
assert client.post("/", json=bad_commit).status_code == 422
good_commit = {"profile":PROFILE,"operation":"commit","evaluationId":"eval-1",
               "inputDigest":response["inputDigest"],
               "receipts":[make_receipt("eval-1",response["inputDigest"],p,accepted=(i==0))
                           for i,p in enumerate(proposals)]}
committed = client.post("/", json=good_commit)
assert committed.status_code == 200, committed.get_data(as_text=True)
out = json.loads(committed.data)
assert out["status"] == "completed"
assert {x["status"] for x in out["outcomes"]} == {"executed","rejected"}
assert client.post("/", json=good_commit).data == committed.data
assert client.post("/", json={**good_commit,"inputDigest":"0"*64}).status_code == 409
assert client.post("/", json={"operation":"bad"}).status_code == 400
assert client.post("/", data="{bad", content_type="application/json").status_code == 400
assert client.post("/", json={**request_body,"evaluationId":"dup","dossiers":[dossiers[0],dossiers[0]]}).status_code == 400
assert client.get("/health").status_code == 200

# Cross-evaluation race: concurrent workers proposing the same stable dossiers
# must converge on exactly the same cached proposals and callIds.
def concurrent_propose(i):
    body = {**request_body, "evaluationId": f"eval-race-{i}"}
    response = app.test_client().post("/", json=body)
    return response.status_code, json.loads(response.data) if response.is_json else {}
with ThreadPoolExecutor(max_workers=6) as pool:
    race_results = list(pool.map(concurrent_propose, range(6)))
assert all(status == 200 for status, _ in race_results), race_results
assert all(result["proposals"] == response["proposals"] for _, result in race_results)

# A dossier with no usable evidence lines is rejected before model work.
empty_evidence = {**request_body, "evaluationId": "eval-no-evidence",
                  "dossiers": [{**dossiers[0], "sources": [{**dossiers[0]["sources"][0], "lines": []}]}]}
assert client.post("/", json=empty_evidence).status_code == 422


# Evaluation isolation: receipts signed with one evaluation's key must not be
# accepted by another evaluation, even when the dossier content/proposals match.
private_b = Ed25519PrivateKey.generate()
public_b = private_b.public_key().public_bytes(
    encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
jwk_b = {"kty": "OKP", "crv": "Ed25519", "x": b64(public_b)}

iso_a_body = {**request_body, "evaluationId": "eval-isolation-a"}
iso_b_body = {**request_body, "evaluationId": "eval-isolation-b",
              "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": jwk_b}}
iso_a_resp = client.post("/", json=iso_a_body)
iso_b_resp = client.post("/", json=iso_b_body)
assert iso_a_resp.status_code == 200, iso_a_resp.get_data(as_text=True)
assert iso_b_resp.status_code == 200, iso_b_resp.get_data(as_text=True)
iso_a = json.loads(iso_a_resp.data)
iso_b = json.loads(iso_b_resp.data)
assert iso_a["inputDigest"] == iso_b["inputDigest"]
assert iso_a["proposals"] == iso_b["proposals"]

# Build receipts for B's evaluation context but sign them with A's private key.
wrong_key_receipts = []
for p in iso_b["proposals"]:
    rec = {"dossierId": p["dossierId"], "callId": p["callId"],
           "action": p["action"], "accepted": True,
           "proposalDigest": proposal_digest(p), "receiptId": "wrong-key-" + p["dossierId"]}
    signed = {"profile": PROFILE, "evaluationId": "eval-isolation-b",
              "inputDigest": iso_b["inputDigest"],
              "receipt": {k: rec[k] for k in
                          ("dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId")}}
    rec["receiptSignature"] = b64(private.sign(canon(signed).encode()))
    wrong_key_receipts.append(rec)
wrong_key_commit = {"profile": PROFILE, "operation": "commit",
                    "evaluationId": "eval-isolation-b", "inputDigest": iso_b["inputDigest"],
                    "receipts": wrong_key_receipts}
assert client.post("/", json=wrong_key_commit).status_code == 422

# The same evaluation succeeds with receipts signed by its own registered key.
def make_receipt_b(eid, digest, proposal, accepted=True):
    rec = {"dossierId": proposal["dossierId"], "callId": proposal["callId"],
           "action": proposal["action"], "accepted": accepted,
           "proposalDigest": proposal_digest(proposal),
           "receiptId": "right-key-" + proposal["dossierId"]}
    signed = {"profile": PROFILE, "evaluationId": eid, "inputDigest": digest,
              "receipt": {k: rec[k] for k in
                          ("dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId")}}
    rec["receiptSignature"] = b64(private_b.sign(canon(signed).encode()))
    return rec

right_key_commit = {**wrong_key_commit,
                    "receipts": [make_receipt_b("eval-isolation-b", iso_b["inputDigest"], p)
                                 for p in iso_b["proposals"]]}
right_key_response = client.post("/", json=right_key_commit)
assert right_key_response.status_code == 200, right_key_response.get_data(as_text=True)

print("all contract-focused local checks passed, including concurrency and evaluation-isolation checks")
