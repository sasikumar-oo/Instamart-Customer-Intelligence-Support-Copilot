"""src/ticket_triage.py -- Support ticket triage pipeline with guardrails, PII redaction, retries, and intelligent routing.
"""
import os
import sys
import re
import json
import time
import random
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
import mock_llm

RAW_DIR = os.path.join("data", "raw")
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Prompt Definitions
ZERO_SHOT_SYSTEM_PROMPT = (
    "You are an Instamart support ticket classifier. Your task is to CLASSIFY the customer issue "
    "into exactly one of four categories: billing, delivery, technical, fraud. "
    "Respond strictly in JSON format with no additional text: "
    '{"category": "<category>", "reason": "<brief justification>"}'
)

FEW_SHOT_SYSTEM_PROMPT = (
    "You are an Instamart support ticket classifier. Your task is to CLASSIFY the customer issue "
    "into exactly one of four categories: billing, delivery, technical, fraud. "
    "Respond strictly in JSON format with no additional text: "
    '{"category": "<category>", "reason": "<brief justification>"}\n\n'
    "EXAMPLES:\n"
    "Ticket: Unauthorized charge on my card for an order I never placed.\n"
    '{"category": "fraud", "reason": "unauthorized card transaction"}\n\n'
    "Ticket: My order was delivered to the wrong address and arrived 40 minutes late.\n"
    '{"category": "delivery", "reason": "wrong address and delayed arrival"}'
)

def redact_pii(text):
    """Apply sequential regex redaction for Email, Phone, and Order ID."""
    # 1. Email: [\w.+-]+@[\w-]+\.[\w.]+ -> [EMAIL]
    email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    # 2. Phone: \b[6-9]\d{9}\b -> [PHONE]
    phone_pattern = re.compile(r"\b[6-9]\d{9}\b")
    # 3. Order ID: ORD-\d{6} -> [ORDER_ID]
    order_pattern = re.compile(r"ORD-\d{6}")

    emails_found = len(email_pattern.findall(text))
    redacted = email_pattern.sub("[EMAIL]", text)

    phones_found = len(phone_pattern.findall(redacted))
    redacted = phone_pattern.sub("[PHONE]", redacted)

    orders_found = len(order_pattern.findall(redacted))
    redacted = order_pattern.sub("[ORDER_ID]", redacted)

    return redacted, emails_found, phones_found, orders_found

def call_mock_with_retry(messages, ticket_id, max_retries=3, base_delay=0.05):
    """Resilient caller wrapping mock_llm.chat with exponential backoff and jitter."""
    for n in range(max_retries):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except mock_llm.TransientLLMError as e:
            wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY] ticket_id={ticket_id}, retry={n}, wait={wait:.4f}s")
            time.sleep(wait)
    # Final attempt
    return mock_llm.chat(messages)

def parse_model_response(response_text):
    """Validate model output across JSON, keyword extraction, and fallback flagging."""
    # Tier 1: Try valid JSON
    try:
        data = json.loads(response_text)
        cat = data.get("category", "").lower().strip()
        if cat in ALLOWED_CATEGORIES:
            return "classified_json", cat
    except Exception:
        pass

    # Tier 2: Extract lowercase words with re.findall(r"[a-z]+", ...)
    words = re.findall(r"[a-z]+", (response_text or "").lower())
    matched_cats = [c for c in ALLOWED_CATEGORIES if c in words]
    if len(matched_cats) == 1:
        return "classified_keyword", matched_cats[0]

    # Tier 3: Invalid JSON or out-of-domain category
    return "flagged_invalid_output", None

def run_triage_pipeline(system_prompt, token_budget=None, is_few_shot=False):
    """Run full triage pipeline on tickets.csv."""
    tickets_path = os.path.join(RAW_DIR, "tickets.csv")
    df_tickets = pd.read_csv(tickets_path, keep_default_na=False)

    mock_llm.reset()

    results = []
    total_tokens_used = 0
    pii_counts = {"email": 0, "phone": 0, "order_id": 0}

    for _, row in df_tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row.get("true_category", "")

        # 1. Input Guardrails
        if raw_text.strip() == "":
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "skipped_empty",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        if mock_llm.est_tokens(raw_text) > 400:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "flagged_long",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # 2. PII Redaction
        redacted_text, e_cnt, p_cnt, o_cnt = redact_pii(raw_text)
        pii_counts["email"] += e_cnt
        pii_counts["phone"] += p_cnt
        pii_counts["order_id"] += o_cnt

        # Assert zero residual PII
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted_text), f"Residual email in {tid}"
        assert not re.search(r"\b[6-9]\d{9}\b", redacted_text), f"Residual phone in {tid}"
        assert not re.search(r"ORD-\d{6}", redacted_text), f"Residual order ID in {tid}"

        # 3. Budget Check
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue

        # 4. Model Call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]

        resp = call_mock_with_retry(messages, tid)
        call_tokens = resp.usage.total_tokens
        total_tokens_used += call_tokens
        resp_content = resp.choices[0].message.content

        # 5. Output Validation
        status, cat = parse_model_response(resp_content)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat or "",
            "true_category": true_cat,
            "tokens_used": call_tokens
        })

    df_results = pd.DataFrame(results)
    return df_results, total_tokens_used, pii_counts

def evaluate_and_route():
    print("================ PART C: SUPPORT TICKET TRIAGE ================\n")

    # 1. PII Redaction Audit
    print("--- 1. PII Redaction & Input Guardrail Verification ---")
    df_raw = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)
    empty_cnt = (df_raw["text"].str.strip() == "").sum()
    long_cnt = (df_raw["text"].apply(mock_llm.est_tokens) > 400).sum()
    print(f"Empty tickets: {empty_cnt} (Reference: 4)")
    print(f"Long tickets (>400 tokens): {long_cnt} (Reference: 4)")

    # 2. Run Zero-Shot Pipeline (No Budget)
    print("\n--- 2. Running Zero-Shot Pipeline (No Budget) ---")
    df_zero, tokens_zero, pii_zero = run_triage_pipeline(ZERO_SHOT_SYSTEM_PROMPT, token_budget=None, is_few_shot=False)
    print(f"PII Substitutions across all tickets: {pii_zero['email']} emails, {pii_zero['phone']} phones, {pii_zero['order_id']} order IDs (Total: {sum(pii_zero.values())}) (Ref: 55, 91, 132, 278)")
    print(f"Status breakdown: {df_zero['status'].value_counts().to_dict()}")

    # Accuracy over classified tickets
    classified_zero = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
    acc_zero = (classified_zero["category"] == classified_zero["true_category"]).mean() * 100
    print(f"Zero-Shot Classified Tickets: {len(classified_zero)} (Ref: 147)")
    print(f"Zero-Shot Accuracy: {acc_zero:.2f}% (Ref: 85.7%)")
    print(f"Zero-Shot Total Tokens: {tokens_zero}, Tokens/Classified: {tokens_zero / len(classified_zero):.1f}")

    print("\nZero-Shot Confusion Matrix (True Category vs Predicted):")
    conf_zero = pd.crosstab(classified_zero["true_category"], classified_zero["category"], margins=True)
    print(conf_zero.to_string())

    # 3. Run Few-Shot Pipeline (No Budget)
    print("\n--- 3. Running Few-Shot Pipeline (No Budget) ---")
    df_few, tokens_few, pii_few = run_triage_pipeline(FEW_SHOT_SYSTEM_PROMPT, token_budget=None, is_few_shot=True)
    classified_few = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])]
    acc_few = (classified_few["category"] == classified_few["true_category"]).mean() * 100
    print(f"Few-Shot Classified Tickets: {len(classified_few)} (Ref: 147)")
    print(f"Few-Shot Accuracy: {acc_few:.2f}% (Ref: 100.0%)")
    print(f"Few-Shot Total Tokens: {tokens_few}, Tokens/Classified: {tokens_few / len(classified_few):.1f}")

    # 4. Token Budget Demo (TOKEN_BUDGET = 6000, Zero-Shot)
    print("\n--- 4. Running Token Budget Demo (TOKEN_BUDGET = 6000) ---")
    df_budget, tokens_budget, _ = run_triage_pipeline(ZERO_SHOT_SYSTEM_PROMPT, token_budget=6000, is_few_shot=False)
    exhausted_cnt = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Budget demo status counts: {df_budget['status'].value_counts().to_dict()}")
    print(f"Budget exhausted tickets: {exhausted_cnt} (Ref: >= 1)")

    # Save outputs/triage_budget_demo.csv
    demo_path = os.path.join(OUTPUTS_DIR, "triage_budget_demo.csv")
    df_budget.to_csv(demo_path, index=False)
    print(f"Saved budget demo output to {demo_path}")

    # 5. Production Run with Routing (Few-Shot, TOKEN_BUDGET = 100000)
    print("\n--- 5. Generating Production Triage Results with Customer Routing ---")
    df_prod, tokens_prod, _ = run_triage_pipeline(FEW_SHOT_SYSTEM_PROMPT, token_budget=100000, is_few_shot=True)

    # Merge customer segments from Part B
    seg_path = os.path.join(OUTPUTS_DIR, "customer_segments.csv")
    df_seg = pd.read_csv(seg_path)
    df_prod = df_prod.merge(df_seg[["customer_id", "is_premium", "segment_name"]], on="customer_id", how="left")
    df_prod["is_premium"] = df_prod["is_premium"].fillna(False).astype(bool)

    # Routing Logic:
    # 1. skipped_empty, flagged_long, flagged_invalid_output, budget_exhausted -> route human_review, priority NORMAL
    # 2. otherwise if category == fraud -> route security_team, priority HIGH
    # 3. otherwise if is_premium == True -> route premium_desk, priority HIGH
    # 4. otherwise -> route support_queue, priority NORMAL
    def assign_routing(row):
        status = row["status"]
        cat = row["category"]
        prem = row["is_premium"]

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            return "human_review", "NORMAL"
        elif cat == "fraud":
            return "security_team", "HIGH"
        elif prem:
            return "premium_desk", "HIGH"
        else:
            return "support_queue", "NORMAL"

    routing_tuples = df_prod.apply(assign_routing, axis=1)
    df_prod["route"] = [r[0] for r in routing_tuples]
    df_prod["priority"] = [r[1] for r in routing_tuples]

    # Category and segment_name left blank when not available / for anomalies
    df_prod.loc[df_prod["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]), "category"] = ""
    df_prod["segment_name"] = df_prod["segment_name"].fillna("")

    prod_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    final_prod = df_prod[prod_cols]

    prod_path = os.path.join(OUTPUTS_DIR, "triage_results.csv")
    final_prod.to_csv(prod_path, index=False)
    print(f"Saved production triage results to {prod_path} ({len(final_prod)} rows)")
    print("\nRouting breakdown by queue and priority:")
    print(final_prod.groupby(["route", "priority"]).size().to_string())

    return {
        "df_zero": df_zero,
        "df_few": df_few,
        "df_budget": df_budget,
        "df_prod": final_prod,
        "tokens_zero": tokens_zero,
        "tokens_few": tokens_few,
        "pii_counts": pii_zero
    }

if __name__ == "__main__":
    evaluate_and_route()
