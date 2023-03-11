import sys
from pathlib import Path

# As PosixPath
sys.path.append(Path(__file__).parent / "lib")
# ^ For finding pygraphistry during development

import time
import traceback
import uuid
import os
import streamlit.components.v1 as components
import streamlit as st
from json import JSONEncoder


_RELEASE = False
_SIMPLE = False

# TODO: inline this
with open("streamlit-style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# How to make a class JSON serializable
# https://stackoverflow.com/a/68926979/3304125
def wrapped_default(self, obj):
    return getattr(obj.__class__, "__json__", wrapped_default.default)(obj)


wrapped_default.default = JSONEncoder().default
JSONEncoder.original_default = JSONEncoder.default
JSONEncoder.default = wrapped_default  # apply the patch


# https://github.com/streamlit/streamlit/issues/653
# https://github.com/streamlit/streamlit/pull/2060
def rerun():
    print(st.experimental_rerun)
    st.experimental_rerun()


## Helpers
####################
def get_now():
    return int(time.time() * 1000)


def gpt_state():
    return st.session_state.gpt_state


def gpt_focused_thought():
    focus = gpt_state().focus
    if focus:
        return gpt_state().thoughts[focus]
    else:
        return None


## Classes
####################
class Thought:
    def __init__(
        self,
        id=str(uuid.uuid4()),
        dataset_id=None,
        parent_id=None,
        prompt="",
        intent="",
        code="",
        updated_at=get_now(),
    ):
        self.id = id
        self.dataset_id = dataset_id
        self.parent_id = parent_id
        self.prompt = prompt
        self.intent = intent
        self.code = code
        self.updated_at = updated_at

    def __repr__(self):
        return str(self.__dict__)

    def __json__(self, **options):
        return self.__dict__


class GptState:
    def __init__(self):
        first_thought = Thought()
        self.focus = first_thought.id
        self.busy = False
        self.action_buffer = []
        self.thoughts = {first_thought.id: first_thought}

    def __repr__(self):
        return str(self.__dict__)

    def __json__(self, **options):
        return self.__dict__


class GptAction:
    def __init__(self, type="unknown", timestamp=0, **kwargs):
        self.type = type
        self.timestamp = timestamp
        self.args = kwargs

    def process(self):
        state = gpt_state()

        try:
            print("Processing action", self.type, self.args)

            if self.type == "fix":
                thought = Thought(**self.args["thought"])
                thought.code = thought.code + " all better now"
                thought.updated_at = self.timestamp
                state.thoughts[thought.id] = thought
            elif self.type == "update":
                thought = Thought(**self.args["thought"])
                state.thoughts[thought.id] = thought
            elif self.type == "focus":
                state.focus = self.args["thought_id"]

            return state
        except Exception:
            print("Error processing action", self)
            traceback.print_exc()
            return state

    def __repr__(self):
        return str(self.__dict__)

    def __json__(self, **options):
        return self.__dict__

    def __str__(self):
        return "[%s] %s" % (self.type, self.args)


## Components
####################
# This call to `declare_component` is the *only thing* you need to do to
# create the binding between Streamlit and your component frontend.
if not _RELEASE:
    # Best practice not to expose to users
    _component_func = components.declare_component(
        "gptistry", url="http://localhost:3001"
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("gptistry", path=build_dir)


def _gptistry(key, type, **kwargs):
    event = _component_func(
        key=key, default=None, type=type, state=gpt_state(), **kwargs
    )
    if event:
        print("GOT Event", event)
        st.session_state.gpt_state.action_buffer.append(GptAction(**event))
        rerun()

    return gpt_state()


def gpt_plot(key, dataset_id, **kwargs):
    return _gptistry(key, "plot", dataset_id=dataset_id, **kwargs)


def gpt_stream(key, root_id, **kwargs):
    return _gptistry(key, "stream", root_id=root_id, **kwargs)


def gpt_thought(key, thought_id, **kwargs):
    return _gptistry(key, "thought", thought_id=thought_id, **kwargs)


def gpt_history(key, **kwargs):
    return _gptistry(key, "history", **kwargs)


def gpt_debug(key, **kwargs):
    return _gptistry(key, "debug", **kwargs)


def gptistry_controller():
    if "gpt_state" not in st.session_state:
        print("initializing gpt_state")
        st.session_state.gpt_state = GptState()

    print("gpt_state", gpt_state())

    action = (
        st.session_state.gpt_state.action_buffer.pop(0)
        if len(st.session_state.gpt_state.action_buffer)
        else None
    )

    if action and st.session_state.gpt_state.busy:
        st.session_state.gpt_state = action.process()
        st.session_state.gpt_state.busy = (
            len(st.session_state.gpt_state.action_buffer) > 0
        )
        rerun()
    elif action:
        # Let the user know we're busy before processing actions
        st.session_state.gpt_state.action_buffer.append(action)
        st.session_state.gpt_state.busy = True
        rerun()

    return action


# During development, run like this:
# app: `$ streamlit run gptistry/__init__.py`
if not _RELEASE and not _SIMPLE:
    current_action = gptistry_controller()
    focus = gpt_focused_thought()

    with st.sidebar:
        st.subheader("GPTistry")
        tab1, tab2, tab3 = st.tabs(["Active Thought", "History", "Debug"])
        with tab1:
            gpt_thought("thought", focus.id)
        with tab2:
            gpt_history("history")
        with tab3:
            if current_action:
                st.markdown(str(current_action))
            else:
                st.markdown("Event loop is empty")
            gpt_debug("debug")

    gpt_stream("stream", focus.id)

    if focus and focus.dataset_id:
        gpt_plot("plot", focus.dataset_id, thought_id=focus.id)

if not _RELEASE and _SIMPLE:
    current_action = gptistry_controller()
    focus = gpt_focused_thought()
    gpt_stream("stream", focus.id)
