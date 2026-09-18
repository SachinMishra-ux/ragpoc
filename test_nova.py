#!/usr/bin/env python3
"""
Standalone Diagnostic Test Script for Amazon Nova Models on AWS Bedrock.
Use this script to verify when AWS has unlocked Bedrock model access on your account.

Usage:
    .venv/bin/python3 test_nova.py
    or
    python3 test_nova.py
"""

import os
import sys
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

print("=" * 70)
print("🔍 AWS Bedrock Amazon Nova Diagnostic & Verification Tool")
print("=" * 70)

# Check credentials in environment
accounts = []

# Account 1
ak1 = os.getenv("AWS_ACCESS_KEY_ID")
sk1 = os.getenv("AWS_SECRET_ACCESS_KEY")
reg1 = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION", "eu-north-1")
token1 = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

if ak1 or token1:
    accounts.append({
        "name": "Account 1 (Primary / sachin)",
        "access_key": ak1,
        "secret_key": sk1,
        "region": reg1,
        "bearer_token": token1,
    })

# Account 2 (Secondary)
ak2 = os.getenv("AWS_ACCESS_KEY_ID2")
sk2 = os.getenv("AWS_SECRET_ACCESS_KEY2")
reg2 = os.getenv("AWS_REGION_2", "us-east-1")

if ak2:
    accounts.append({
        "name": "Account 2 (Secondary / Previous AWS Account)",
        "access_key": ak2,
        "secret_key": sk2,
        "region": reg2,
        "bearer_token": None,
    })

if not accounts:
    print("❌ Error: No AWS credentials found in .env.")
    sys.exit(1)

for idx, acc in enumerate(accounts, start=1):
    print(f"[{acc['name']}]")
    print(f"  • Region       : {acc['region']}")
    print(f"  • Access Key   : {'Present (' + acc['access_key'][:8] + '...)' if acc['access_key'] else 'Missing'}")
    print(f"  • Secret Key   : {'Present' if acc['secret_key'] else 'Missing'}")
    print(f"  • Bearer Token : {'Present' if acc['bearer_token'] else 'None'}")
print("-" * 70)

successes = []
failures = []

def test_via_langchain(model_id: str, region: str, access_key: str, secret_key: str, bearer_token: str, acc_label: str):
    """Tests model invocation using langchain-aws ChatBedrockConverse."""
    from langchain_aws import ChatBedrockConverse

    auth_desc = "Bearer Token" if bearer_token else "IAM Keys"
    print(f"\n▶ [{acc_label}] Testing {model_id} in {region} via LangChain [{auth_desc}]...")

    kwargs = {
        "model": model_id,
        "region_name": region,
        "temperature": 0.2,
    }
    if bearer_token:
        kwargs["bedrock_api_key"] = bearer_token
    elif access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    else:
        print("  ⏭ Skipped: Incomplete credentials.")
        return False

    saved_token = os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    try:
        llm = ChatBedrockConverse(**kwargs)
        response = llm.invoke("Hello! In one short sentence, who are you and what model are you?")
        reply = response.content if isinstance(response.content, str) else str(response.content)
        print(f"  ✅ SUCCESS!")
        print(f"  💬 Reply: {reply.strip()[:180]}...")
        successes.append((acc_label, model_id, region, auth_desc))
        return True
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e).strip()
        print(f"  ❌ FAILED: [{err_type}] {err_msg[:220]}")
        failures.append((acc_label, model_id, region, auth_desc, err_type, err_msg))
        return False
    finally:
        if saved_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = saved_token


def test_via_boto3(model_id: str, region: str, access_key: str, secret_key: str, acc_label: str):
    """Tests model invocation using raw boto3 converse API."""
    import boto3

    print(f"\n▶ [{acc_label}] Testing {model_id} in {region} via Raw boto3...")
    saved_token = os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        resp = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Hello! Reply with 'OK'."}]}],
        )
        out_msg = resp.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        print(f"  ✅ SUCCESS via boto3!")
        print(f"  💬 Response: {out_msg.strip()}")
        successes.append((acc_label, model_id, region, "boto3"))
        return True
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e).strip()
        print(f"  ❌ FAILED via boto3: [{err_type}] {err_msg[:220]}")
        failures.append((acc_label, model_id, region, "boto3", err_type, err_msg))
        return False
    finally:
        if saved_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = saved_token


def main():
    for acc in accounts:
        acc_name = acc["name"]
        reg = acc["region"]
        ak = acc["access_key"]
        sk = acc["secret_key"]
        token = acc["bearer_token"]

        models_to_test = [
            ("amazon.nova-2-lite-v1:0", reg),
            ("us.amazon.nova-2-lite-v1:0", "us-east-1"),
            ("amazon.nova-lite-v1:0", reg),
            ("us.amazon.nova-lite-v1:0", "us-east-1"),
        ]

        print(f"\n{'='*70}\nTesting {acc_name}\n{'='*70}")

        # Test with bearer token if present
        if token:
            for m, r in models_to_test:
                ok = test_via_langchain(m, r, ak, sk, token, acc_name)
                if ok:
                    break

        # Test with IAM keys
        if ak and sk:
            for m, r in models_to_test:
                ok = test_via_langchain(m, r, ak, sk, None, acc_name)
                if ok:
                    break
            test_via_boto3(models_to_test[0][0], models_to_test[0][1], ak, sk, acc_name)

    # Final Summary
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if successes:
        print("🎉 GREAT NEWS! Working Amazon Nova configuration found:")
        for acc_lbl, m, r, auth in successes:
            print(f"   ✔ [{acc_lbl}] Model: {m} in {r} ({auth})")
        print("\n👉 To set this working account as primary for the web app:")
        print("   Update AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env with these working keys.")
    else:
        print("⚠️ Neither AWS account could invoke Amazon Nova yet.")
        for acc_lbl, m, r, auth, err_t, err_m in failures:
            if "InvalidClientTokenId" in err_t or "The security token included in the request is invalid" in err_m:
                print(f"   • [{acc_lbl}] Authentication Failed: The Access Key ID is unrecognized or inactive in AWS IAM.")
            elif "ValidationException" in err_t or "Operation not allowed" in err_m:
                print(f"   • [{acc_lbl}] Bedrock Account Hold: AWS returned 'ValidationException: Operation not allowed'.")
        print("\n👉 Suggestions:")
        print("   1. For Account 2: Check in AWS Console -> IAM -> Security Credentials to make sure the Access Key ID is 'Active' and typed correctly.")
        print("   2. For Account 1: Wait for AWS Support to lift the Bedrock hold.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
