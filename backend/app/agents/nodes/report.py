from langchain_groq import ChatGroq
from app.agents.state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from langchain_core.output_parsers import StrOutputParser
import re

def report_node(state: AgentState) -> dict:
    """
    Report Agent: Generates a professional, structured clinical triage report.
    """
    urgency = state.get("urgency_level", "Unknown")
    dept = state.get("recommended_department", "Unknown")
    reasoning = state.get("analysis_reasoning", "")
    apt_status = state.get("appointment_status", "")
    symptoms = state.get("symptoms", "Not specified")
    duration = state.get("duration", "Not specified")
    severity = state.get("severity", "Not specified")

    urgency_badge = {"High": "🔴 HIGH PRIORITY", "Medium": "🟡 MEDIUM PRIORITY", "Low": "🟢 LOW PRIORITY"}.get(urgency, "⚪ UNDER REVIEW")

    try:
        llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="openai/gpt-oss-120b", temperature=0.2)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the VitalGate AI Chief Medical Reporting Agent. "
             "Generate a highly concise, professional clinical triage report. "
             "Output EXACTLY this format and NO other text. Do NOT add borders or extra text:\n\n"
             "**URGENCY STATUS:** {urgency_badge}\n"
             "**Recommended Department:** {dept}\n\n"
             "**📋 PATIENT PRESENTATION**\n"
             "- **Primary Complaint:** [1 sentence summary: {symptoms}]\n"
             "- **Duration:** {duration}\n"
             "- **Severity:** {severity}\n\n"
             "**🔬 CLINICAL ASSESSMENT**\n"
             "[Write EXACTLY ONE concise, non-alarming sentence summarizing the condition based on: {reasoning}]\n\n"
             "**✅ NEXT STEPS**\n"
             "1. [Brief immediate action]\n"
             "2. [Brief monitoring instruction]\n"
             "3. [Brief emergency warning sign]"
            ),
            ("human", "Generate the triage report now.")
        ])

        chain = prompt | llm | StrOutputParser()

        summary = chain.invoke({
            "urgency_badge": urgency_badge,
            "dept": dept,
            "symptoms": symptoms,
            "duration": duration,
            "severity": severity,
            "reasoning": reasoning,
            "apt_status": apt_status
        })
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()

    except Exception as e:
        print(f"Report node error: {str(e)}")
        raise RuntimeError(f"Failed to generate final report: {str(e)}")

    return {
        "final_summary": summary,
        "messages": [AIMessage(content=summary)]
    }
