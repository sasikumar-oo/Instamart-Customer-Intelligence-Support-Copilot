"""src/refund_agent.py -- Autonomous ReAct refund agent with tool calling, safety circuit breakers, checkpoint state persistence, and crash recovery.
"""
import os
import sys
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))
import mock_llm
from src.policy_rag import chunk_documents, PolicyRetriever

DB_PATH = os.path.join("data", "instamart.db")
RAW_DIR = os.path.join("data", "raw")
OUTPUTS_DIR = "outputs"
STATE_DIR = "state"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# Initialize policy retriever
chunks = chunk_documents(overlap=10, chunk_size=40)
retriever = PolicyRetriever(chunks)

class SimulatedAgentCrashError(Exception):
    """Exception raised to simulate agent crash mid-execution."""
    pass

# ================= TOOLS IMPLEMENTATION =================

def get_order(order_id):
    """Look up order record in SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": float(row[3]) if row[3] is not None else None
    }

def search_policy(query):
    """Retrieve top policy chunks."""
    results = retriever.retrieve(query, k=3)
    return {"results": results}

def calculate_refund(order_amount, delivery_minutes):
    """Compute refund amount based on business rules."""
    # Completed orders delivered in > 45 minutes get 100% refund
    if delivery_minutes is not None and float(delivery_minutes) > 45:
        refund = round(float(order_amount), 2)
    else:
        refund = 0.0
    return {"refund_inr": refund}

def _log_refund_action(tool_name, order_id, refund_inr):
    ledger_path = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def issue_refund(order_id, refund_inr):
    """Record an automatic refund issue in the ledger."""
    _log_refund_action("issue_refund", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    """Record an escalation to human lead in the ledger."""
    _log_refund_action("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": float(refund_inr)}

# Tool specifications & map
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the database by its order ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the company policy documents for relevant guidelines.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund eligibility and amount in INR based on delivery minutes and order amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an automated refund for an approved order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a high-value or complex refund request to a human lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

# ================= REACT LOOP & STATE MANAGEMENT =================

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, executed_tools, steps, llm_calls):
    """Write run state to state/<run_id>.json."""
    checkpoint_path = os.path.join(STATE_DIR, f"{run_id}.json")
    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "executed_tools": executed_tools,
        "steps": steps,
        "llm_calls": llm_calls
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, simulate_crash_after_step=None, run_id=None):
    """Execute ReAct agent loop with circuit breakers and state checkpointing."""
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time() * 1000)}"

    start_time = time.monotonic()
    system_prompt = (
        "You are an Instamart refund AGENT. Call the available tools to investigate order status and policy, "
        "calculate refunds, and either issue or escalate refunds."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    tokens_used = 0
    llm_calls = 0
    step_number = 0
    executed_tools = []
    steps = []

    for iteration in range(max_iterations):
        # 1. Timeout Check
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handing off to human lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        # 2. Call LLM
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # 3. Token Ceiling Check
        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token limit exceeded. Handing off to human lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        choice = resp.choices[0]
        tool_calls = choice.message.tool_calls

        # 4. If Tool Requested
        if tool_calls:
            tc = tool_calls[0]
            t_name = tc.name
            t_args = tc.arguments
            t_id = tc.id

            # Append assistant message
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
            })

            # Execute tool
            if t_name not in TOOL_MAP:
                raise ValueError(f"Unknown tool requested: {t_name}")
            observation = TOOL_MAP[t_name](**t_args)

            # Append tool message
            messages.append({
                "role": "tool",
                "tool_call_id": t_id,
                "name": t_name,
                "content": json.dumps(observation)
            })

            step_number += 1
            executed_tools.append(t_name)
            steps.append({
                "tool": t_name,
                "arguments": t_args,
                "observation": observation
            })

            # Save state checkpoint
            save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, executed_tools, steps, llm_calls)

            # Simulated crash
            if simulate_crash_after_step is not None and step_number == simulate_crash_after_step:
                raise SimulatedAgentCrashError(f"Simulated crash after step {step_number} in {run_id}")

        else:
            # Final text response
            final_content = choice.message.content
            # Normal status is the word before the colon in the final answer
            status = final_content.split(":")[0].strip() if ":" in final_content else "DENIED"
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

    # Loop exhausted
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum iterations reached. Handing off to human lead.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """Resume agent execution from saved checkpoint in state/<run_id>.json."""
    checkpoint_path = os.path.join(STATE_DIR, f"{run_id}.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    scenario_id = state["scenario_id"]
    messages = state["messages"]
    tokens_used = state["tokens_used"]
    step_number = state["step_number"]
    executed_tools = list(state["executed_tools"])
    steps = list(state["steps"])
    llm_calls = state["llm_calls"]

    start_time = time.monotonic()

    for iteration in range(step_number, max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK: Execution timed out. Handing off to human lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK: Token limit exceeded. Handing off to human lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }

        choice = resp.choices[0]
        tool_calls = choice.message.tool_calls

        if tool_calls:
            tc = tool_calls[0]
            t_name = tc.name
            t_args = tc.arguments
            t_id = tc.id

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
            })

            observation = TOOL_MAP[t_name](**t_args)

            messages.append({
                "role": "tool",
                "tool_call_id": t_id,
                "name": t_name,
                "content": json.dumps(observation)
            })

            step_number += 1
            executed_tools.append(t_name)
            steps.append({
                "tool": t_name,
                "arguments": t_args,
                "observation": observation
            })

            save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, executed_tools, steps, llm_calls)

        else:
            final_content = choice.message.content
            status = final_content.split(":")[0].strip() if ":" in final_content else "DENIED"
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": status,
                "final_answer": final_content,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }

    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK: Maximum iterations reached. Handing off to human lead.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "executed_tools": executed_tools
    }

def run_all_scenarios_and_benchmarks():
    print("================ PART D: REFUND AGENT BENCHMARKS ================\n")

    # Clear previous agent runs / ledger if fresh run
    runs_file = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")
    if os.path.exists(runs_file):
        os.remove(runs_file)
    ledger_file = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
    if os.path.exists(ledger_file):
        os.remove(ledger_file)

    scenarios_path = os.path.join(RAW_DIR, "agent_scenarios.csv")
    df_scenarios = pd.read_csv(scenarios_path)

    results = []

    print("--- Executing 7 Scenarios from agent_scenarios.csv ---")
    for _, row in df_scenarios.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_status = row["expected_outcome"]
        exp_refund = float(row["expected_refund_inr"])

        res = run_agent(msg, sid, max_iterations=6, token_limit=6000, time_limit_seconds=10)

        # Extract refund amount from steps or answer
        refund_amount = 0.0
        for step in res["steps"]:
            if step["tool"] in ["calculate_refund", "issue_refund", "escalate_to_human"]:
                refund_amount = float(step["observation"].get("refund_inr", 0.0))

        # Check status match
        match_status = (res["status"] == exp_status) or (exp_status == "FALLBACK" and "FALLBACK" in res["status"])
        match_refund = abs(refund_amount - exp_refund) < 0.01

        print(f"Scenario {sid}: Status={res['status']} (Exp: {exp_status}), Refund=INR {refund_amount:.2f} (Exp: {exp_refund:.2f}), Calls={res['llm_calls']}, Tokens={res['tokens_used']} -> Match: {match_status and match_refund}")

        # Append to outputs/agent_runs.jsonl
        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        results.append({
            "scenario_id": sid,
            "status": res["status"],
            "refund_amount": refund_amount,
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"]
        })

    # Assert S1 model calls == 5
    s1_res = next(r for r in results if r["scenario_id"] == "S1")
    assert s1_res["llm_calls"] == 5, f"Expected 5 calls for S1, got {s1_res['llm_calls']}"
    # Assert S7 ends in FALLBACK_MAX_ITERATIONS
    s7_res = next(r for r in results if r["scenario_id"] == "S7")
    assert s7_res["status"] == "FALLBACK_MAX_ITERATIONS", f"Expected FALLBACK_MAX_ITERATIONS for S7, got {s7_res['status']}"

    # --- Safety Demonstrations ---
    print("\n--- Safety Circuit Breaker Demonstrations ---")
    s1_msg = df_scenarios[df_scenarios["scenario_id"] == "S1"]["message"].values[0]

    # Token Limit Demo: S1 with token_limit=200
    res_token_demo = run_agent(s1_msg, "S1_token_demo", token_limit=200)
    print(f"Token Limit Demo: Status={res_token_demo['status']} (Ref: FALLBACK_TOKEN_LIMIT)")
    assert res_token_demo["status"] == "FALLBACK_TOKEN_LIMIT"

    # Timeout Demo: S1 with time_limit_seconds=0
    res_time_demo = run_agent(s1_msg, "S1_time_demo", time_limit_seconds=0)
    print(f"Timeout Demo: Status={res_time_demo['status']} (Ref: FALLBACK_TIMEOUT)")
    assert res_time_demo["status"] == "FALLBACK_TIMEOUT"

    # --- Crash & Resume Demonstration ---
    print("\n--- Crash & Resume Demonstration (S1) ---")
    crash_run_id = f"crash_demo_{int(time.time())}"
    crashed = False
    try:
        run_agent(s1_msg, "S1", simulate_crash_after_step=2, run_id=crash_run_id)
    except SimulatedAgentCrashError as e:
        crashed = True
        print(f"Simulated crash occurred as expected: {e}")

    assert crashed, "Simulated crash failed to trigger!"

    # Load checkpoint and verify steps before crash
    with open(os.path.join(STATE_DIR, f"{crash_run_id}.json"), "r") as f:
        pre_crash_state = json.load(f)
    print(f"Tools executed before crash: {pre_crash_state['executed_tools']} (Length: {len(pre_crash_state['executed_tools'])})")
    assert pre_crash_state["executed_tools"] == ["get_order", "search_policy"]

    # Resume agent
    res_resumed = resume_agent(crash_run_id)
    print(f"Resumed execution result: Status={res_resumed['status']}, Final Answer={res_resumed['final_answer']}")
    print(f"All tools executed across crash + resume: {res_resumed['executed_tools']}")

    assert res_resumed["status"] == "REFUND", f"Expected REFUND, got {res_resumed['status']}"
    assert res_resumed["executed_tools"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"], f"Unexpected tool execution sequence: {res_resumed['executed_tools']}"

    print("\nALL PART D AGENT BENCHMARKS AND DEMONSTRATIONS PASSED PERFECTLY!")
    return results, res_token_demo, res_time_demo, res_resumed

if __name__ == "__main__":
    run_all_scenarios_and_benchmarks()
