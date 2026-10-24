"""MOCK_MODEL=1 python test_local.py  (no model calls)"""
import os, json, base64
os.environ["MOCK_MODEL"] = "1"; os.environ["DATABASE_URL"] = "sqlite:///t.db"
if os.path.exists("t.db"): os.remove("t.db")
import app as A
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
c = A.app.test_client()
sk = Ed25519PrivateKey.generate()
raw = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
x = base64.urlsafe_b64encode(raw).decode().rstrip("=")
def dos(i, p="stable_core"):
    return {"dossierId": f"d{i}", "partition": p, "receivedAt": "t", "mailbox": "m", "objective": "o",
            "sources": [{"sourceId": "s1", "kind": "email", "provenance": "external", "title": "t",
                         "lines": [{"lineId": f"d{i}-l1", "text": "hello"}, {"lineId": f"d{i}-l2", "text": "bye"}]}]}
def req(eid, ds):
    return {"profile": A.PROFILE, "operation": "propose", "evaluationId": eid,
            "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": x}},
            "allowedActions": A.ALLOWED_ACTIONS,
            "corpus": {"coreId": "c", "auditId": "a", "stableCount": 3, "freshCount": 1}, "dossiers": ds}
ds = [dos(1), dos(2), dos(3, "fresh_audit")]
r1 = c.post("/", json=req("e1", ds)); assert r1.status_code == 200, r1.data
assert c.post("/", json=req("e1", ds)).data == r1.data
assert c.post("/", json=req("e1", ds[:2])).status_code == 409
p = r1.get_json()
# failed model -> fallback must NOT be cached
A.call_model = lambda d, dl: []
ds2 = [dos(7), dos(8)]
r2 = c.post("/", json=req("e2", ds2)); assert r2.status_code == 200
with A.engine.begin() as conn:
    n = conn.execute(A.text("SELECT COUNT(*) FROM decisions WHERE dossier_id IN ('d7','d8')")).scalar()
assert n == 0, "fallbacks were cached"
# commit with signed receipts
def receipt(prop, eid, digest, ok=True, bad=False):
    r = {"dossierId": prop["dossierId"], "callId": prop["callId"], "action": prop["action"], "accepted": ok,
         "proposalDigest": A.proposal_digest(prop), "receiptId": "r-" + prop["dossierId"]}
    msg = {"profile": A.PROFILE, "evaluationId": eid, "inputDigest": digest, "receipt": r}
    sig = sk.sign(A.canon(msg).encode())
    r["receiptSignature"] = base64.urlsafe_b64encode(b"x" * 64 if bad else sig).decode().rstrip("=")
    return r
cm = lambda rs: {"profile": A.PROFILE, "operation": "commit", "evaluationId": "e1", "inputDigest": p["inputDigest"], "receipts": rs}
bad = [receipt(q, "e1", p["inputDigest"], bad=(i == 0)) for i, q in enumerate(p["proposals"])]
assert c.post("/", json=cm(bad)).status_code == 422
good = cm([receipt(q, "e1", p["inputDigest"]) for q in p["proposals"]])
a = c.post("/", json=good); assert a.status_code == 200 and a.get_json()["status"] == "completed"
assert c.post("/", json=good).data == a.data
print("all local checks passed")
