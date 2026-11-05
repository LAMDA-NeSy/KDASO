GENERATE_PROMPT = '''
Please strictly adhere to the following instructions to complete the task. This task is **format-constrained reasoning**; output format errors will result in system parsing failure.

# Task Objectives
Based on the current liability adjudication case and 4 historical similar cases, generate two segments of structured content:
1. Inductive analysis of the liability adjudication logic of similar cases;
2. Optimization recommendations for the verification method regarding the current case.

# Mandatory Output Specifications (❗ Violation of any clause will be deemed invalid output)
1. **Only the following two XML tags are permitted**, and they must appear in pairs without nesting:
   - `<judge> ... </judge>`
   - `<advice> ... </advice>`
2. **Generation of any other XML/HTML tags is prohibited**, including but not limited to:
   `<rule>`, `<reason>`, `<summary>`, `<step>`, `<explanation>`, `<details>`, `<conclusion>`, `<analysis>`, etc.
3. **Outputting any content outside the tags is prohibited** (including explanations, titles, empty lines, comments, etc.). Output must begin with `<judge>` and end with `</advice>`.
4. **Only plain text is permitted within tags**; inline attributes (e.g., `<judge id="1">`) or self-closing tags (e.g., `<br/>`) are prohibited.

# Content Requirements
## <judge> Section
- Focus of analysis: How to derive the "true adjudication result" through factual chain reasoning, starting from "order content" and based on the rules within the "verification method";
- Must embody: Triple logic of spatiotemporal consistency verification, cross-validation of evidence, and mapping of rule clauses.

## <advice> Section
- Output only: The **verification method text** to be adopted for the current case (i.e., rewriting or addition of the "Definition" field in the rule knowledge base);
- Format: Begin with "- Definition:", followed by a description of rigorous and actionable verification steps;
- Inclusion of justification reasons, case citations, or meta-commentary (e.g., "It is recommended to modify to...") is prohibited.

# Example (Correct Format)
<judge>In similar cases, when the driver reports "passenger late", combined with no passenger communication records in the vehicle audio, the possibility of driver misjudgment is excluded; ultimately, liability is assigned to the passenger based on Article 3.2 of the "Driver-Passenger Performance Specification".</judge>
<advice>- Definition: After the driver arrives at the pickup point, record the waiting duration via the platform timer; if there is no valid communication evidence (such as call recordings, text messages), it is deemed as passenger lateness.</advice>
'''

PROBLEM_WITH_EXPERIENCE_TEMPLATE="""
You are tasked with addressing the following problem:
{problem}
In the course of resolving this issue, it is imperative that you first thoroughly examine and comprehend the following pertinent guidelines and experiential knowledge:
{experiences}
"""

REFINE_PROMPT_1='''
## Task Objective:
You are a professional **General Liability Adjudication Expert**. Your current task is to handle the given dispute case.

Based on the provided **[Factual Information]** and the identified **[Key Events/Anomalies]**, strictly evaluate each **Adjudication Rule** sequentially to determine if it applies.
1. First, evaluate the high-priority rules (the first N rules, depending on the rule list). If any rule is matched, immediately output the result.
2. If none of the high-priority rules are matched, proceed to evaluate each **[Key Event/Anomaly]** individually using the remaining rules.
3. Finally, output the conclusion based on the matched rule with the **highest priority**.

**Important Notes:**
- Adjudication rules are numbered starting from **1**; a lower number indicates **higher priority**.
- The adjudication process must strictly follow the priority order. Once a rule is matched, **immediately stop** further evaluation and output the result.
- If **[Key Events/Anomalies]** are empty or missing, output "No Liability" based on the clause regarding "No Anomalies" in the rules (typically Rule 1 or similar).
- If critical evidence (e.g., recordings, logs, surveillance) is missing, directly output "Unable to Determine" or the corresponding conclusion based on the clause regarding "Missing Evidence".
- For specific conditions involved in the rules (e.g., reporting reasons, time thresholds, amount discrepancies), **strict matching** is required. Fuzzy matching or failure to meet specific limitations (e.g., mentioning only "traffic congestion" without "road closure/accident") shall be considered as **not matched**.
- Evaluate the adjudication rules sequentially; **stop immediately upon a match**.

## [Adjudication Rules] (Priority from High to Low):
'''

REFINE_PROMPT_2='''
## Output Requirements:

1. **Liability Determination Result**
   Output Format: `<result>No Liability</result>`

2. **Matched Rule Description**
   Output Format: `<rule>Actual cost did not exceed the estimated price or the fixed price was not exceeded</rule>`

3. **Specific Analysis Explanation**
   Output Format: `<analysis>Since the driver mentioned xx at the deviation point and the passenger did not modify the route, Rule 9 is matched.</analysis>`

### **Final Output Example**:
   ```
   <result>No Liability</result>  
   <rule>Actual cost did not exceed the estimated price or the fixed price was not exceeded</rule>  
   <analysis>Since the driver mentioned xx at the deviation point and the passenger did not modify the route, Rule 9 is matched.</analysis>
   ```
'''