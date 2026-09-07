import streamlit as st
import uuid

from agent.employee_agent import ask_employee


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Blue Orbit Office Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}


/* ==========================================================
   MAIN HEADER
   ========================================================== */

.main-header {
    padding: 5px 0 20px 0;
}

.main-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 5px;
}

.main-subtitle {
    font-size: 15px;
    color: #888;
    margin-bottom: 10px;
}

.online-badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    background-color: #e8f7ee;
    color: #16803c;
    font-size: 13px;
    font-weight: 600;
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    border-right: 1px solid #333;
}

.sidebar-title {
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 18px;
}

.employee-card {
    padding: 14px;
    border-radius: 12px;
    background-color: #f3f3f3;
    margin-top: 8px;
    margin-bottom: 18px;
}

.employee-label {
    font-size: 11px;
    color: #777;
    margin-bottom: 4px;
}

.employee-id {
    font-size: 17px;
    font-weight: 700;
    color: #222;
}


/* ==========================================================
   WELCOME CARD
   ========================================================== */

.welcome-card {
    padding: 25px;
    border-radius: 16px;
    background-color: #f7f8fa;
    border: 1px solid #e4e4e4;
    margin-top: 10px;
    margin-bottom: 22px;
}

.welcome-title {
    font-size: 22px;
    font-weight: 650;
    margin-bottom: 8px;
    color: #222;
}

.welcome-text {
    color: #666;
    font-size: 14px;
    line-height: 1.6;
}


/* ==========================================================
   SUGGESTIONS
   ========================================================== */

.suggestion-title {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 10px;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;
    color: #999;
    font-size: 12px;
    padding: 25px 0 5px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"

if "employee_id" not in st.session_state:
    st.session_state.employee_id = "EMP001"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def start_new_chat():
    """Start a completely new conversation."""

    st.session_state.messages = []
    st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"
    st.session_state.pending_question = None


def clear_current_chat():
    """Clear the current conversation."""

    st.session_state.messages = []
    st.session_state.pending_question = None


def update_chat_history():
    """Add or update the current conversation."""

    if not st.session_state.messages:
        return

    first_question = next(
        (
            message["content"]
            for message in st.session_state.messages
            if message["role"] == "user"
        ),
        "New Conversation",
    )

    title = first_question[:35]

    if len(first_question) > 35:
        title += "..."

    existing_threads = [
        chat["thread_id"]
        for chat in st.session_state.chat_history
    ]

    # Add new conversation
    if st.session_state.thread_id not in existing_threads:

        st.session_state.chat_history.insert(
            0,
            {
                "title": title,
                "thread_id": st.session_state.thread_id,
                "messages": list(st.session_state.messages),
            },
        )

    # Update existing conversation
    else:

        for chat in st.session_state.chat_history:

            if chat["thread_id"] == st.session_state.thread_id:

                chat["messages"] = list(
                    st.session_state.messages
                )

                break


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRANDING
    # --------------------------------------------------------

    st.markdown(
        """
<div class="sidebar-title">
🤖 Blue Orbit Office Assistant
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # EMPLOYEE
    # --------------------------------------------------------

    st.markdown("### 👤 Employee")

    employee_id = st.text_input(
        "Employee ID",
        value=st.session_state.employee_id,
        placeholder="Example: EMP001",
        label_visibility="collapsed",
    )

    employee_id = employee_id.strip().upper()

    st.session_state.employee_id = employee_id

    # IMPORTANT:
    # HTML is intentionally NOT indented inside this string.
    # This prevents Streamlit from treating it as a code block.

    st.markdown(
        f"""
<div class="employee-card">
<div class="employee-label">
CURRENT EMPLOYEE
</div>
<div class="employee-id">
{employee_id}
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕  New Chat",
        use_container_width=True,
    ):

        update_chat_history()
        start_new_chat()
        st.rerun()

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑️  Clear Current Chat",
        use_container_width=True,
    ):

        clear_current_chat()
        st.rerun()

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    st.markdown("### 🕘 Chat History")

    if not st.session_state.chat_history:

        st.caption(
            "Previous conversations will appear here."
        )

    else:

        for index, chat in enumerate(
            st.session_state.chat_history
        ):

            if st.button(
                f"💬 {chat['title']}",
                key=f"history_{index}",
                use_container_width=True,
            ):

                st.session_state.messages = list(
                    chat["messages"]
                )

                st.session_state.thread_id = chat[
                    "thread_id"
                ]

                st.rerun()

    # --------------------------------------------------------
    # SUPPORTED TOPICS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### 🧩 Supported Topics")

    st.markdown(
        """
**👤 Employee**
- Employee details
- Leave balance
- Expenses

**💻 IT**
- IT assets
- Assigned laptop

**📚 Policies**
- Leave policy
- WFH policy
- Attendance policy

**🏢 Office**
- Office information
"""
    )

    st.markdown("---")

    st.caption(
        "LangGraph • Qwen3 • RAG"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
<div class="main-header">
<div class="main-title">
🤖 Blue Orbit Office Assistant
</div>
<div class="main-subtitle">
Your AI-powered workplace assistant
</div>
<div class="online-badge">
● Agent Online
</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
<div class="welcome-card">
<div class="welcome-title">
👋 Hey! How can I help you?
</div>
<div class="welcome-text">
Ask me about your employee information,
leave balance, expenses, IT assets,
office details, or company policies.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="suggestion-title">💡 Try asking</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SUGGESTED QUESTIONS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🏖️  What is my leave balance?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What is my leave balance?"
            )

            st.rerun()

    with col2:

        if st.button(
            "📜  What is the leave policy?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What is the leave policy?"
            )

            st.rerun()

    with col3:

        if st.button(
            "💻  What laptop was assigned to me?",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                "What laptop was assigned to me?"
            )

            st.rerun()


# ============================================================
# DISPLAY EXISTING CONVERSATION
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask your Office Assistant..."
)


# ============================================================
# HANDLE SUGGESTED QUESTION
# ============================================================

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    # --------------------------------------------------------
    # VALIDATE EMPLOYEE ID
    # --------------------------------------------------------

    if not employee_id:

        st.warning(
            "Please enter your Employee ID in the sidebar."
        )

        st.stop()

    # --------------------------------------------------------
    # ADD USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # CALL AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                answer = ask_employee(
                    question=question,
                    employee_id=employee_id,
                    thread_id=st.session_state.thread_id,
                )

                if not answer:

                    answer = (
                        "Sorry, I couldn't generate a response."
                    )

                # Display answer
                st.markdown(answer)

                # Save assistant message
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # Update sidebar history
                update_chat_history()

            except Exception as e:

                st.error(
                    "Something went wrong while "
                    "processing your request."
                )

                st.caption(
                    f"Technical details: {str(e)}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
Blue Orbit Office Assistant • Internal Workplace Assistant
</div>
""",
    unsafe_allow_html=True,
)