import re
from typing import Optional, Dict, Any
from enum import Enum

class Intent(str, Enum):
    START_EXPERIMENT = "START_EXPERIMENT"
    NEXT_STEP = "NEXT_STEP"
    CHECK_CURRENT_ACTION = "CHECK_CURRENT_ACTION"
    WHAT_AM_I_DOING = "WHAT_AM_I_DOING"
    IDENTIFY_OBJECT = "IDENTIFY_OBJECT"
    IDENTIFY_COLOR = "IDENTIFY_COLOR"
    WHAT_MOVEMENTS = "WHAT_MOVEMENTS"
    REPEAT_STEP = "REPEAT_STEP"
    PAUSE_EXPERIMENT = "PAUSE_EXPERIMENT"
    RESUME_EXPERIMENT = "RESUME_EXPERIMENT"
    RESET_EXPERIMENT = "RESET_EXPERIMENT"
    STOP_EXPERIMENT = "STOP_EXPERIMENT"
    SNAPSHOT = "SNAPSHOT"
    RECORD_START = "RECORD_START"
    RECORD_STOP = "RECORD_STOP"
    HELP = "HELP"
    UNKNOWN = "UNKNOWN"

class CommandParser:
    def __init__(self):
        self.intent_patterns = [
            (Intent.START_EXPERIMENT, [
                r"\bstart( the)? experiment\b",
                r"\bstart\b",
                r"\bbegin( the)? experiment\b",
                r"\blaunch experiment\b"
            ]),
            (Intent.WHAT_AM_I_DOING, [
                r"\bwhat( am i| is being| is) doing\b",
                r"\bwhat is happening\b",
                r"\bwhat is going on\b",
                r"\bwhat('?s| is) my action\b",
                r"\bcurrent action\b",
                r"\bdescribe( my)? action\b",
                r"\bwhat action is (going on|detected)\b",
                r"\bwhat am i doing right now\b"
            ]),
            (Intent.IDENTIFY_OBJECT, [
                r"\bwhat is (this|that) object\b",
                r"\bwhat object( is this| is that)?\b",
                r"\bwhat am i holding\b",
                r"\bwhat('?s| is) in my hand\b",
                r"\bidentify (this|the) object\b",
                r"\bdetect object\b",
                r"\bwhich item is this\b",
                r"\bwhat items? do you see\b",
                r"\bwhat objects? do you see\b"
            ]),
            (Intent.IDENTIFY_COLOR, [
                r"\bwhat colou?r is (this|that|it)\b",
                r"\bwhich colou?r is (this|that|it)\b",
                r"\bwhat is the colou?r\b",
                r"\bcolou?r of (this|the) object\b",
                r"\btell me the colou?r\b",
                r"\bidentify (the )?colou?r\b",
                r"\bwhat colou?r\b",
                r"\bwhich colou?r\b"
            ]),
            (Intent.WHAT_MOVEMENTS, [
                r"\bwhat are my movements?\b",
                r"\bwhat movement\b",
                r"\btrack movement\b",
                r"\bhow am i moving\b",
                r"\bdetect movement\b",
                r"\bposture and movement\b",
                r"\bwhat is my posture\b",
                r"\bmovement analysis\b"
            ]),
            (Intent.NEXT_STEP, [
                r"\b(what('?s| is) )?(the )?next step\b",
                r"\btell me (the )?next step\b",
                r"\bwhat do i do( next)?\b",
                r"\bnext procedure\b",
                r"\bguidance\b"
            ]),
            (Intent.CHECK_CURRENT_ACTION, [
                r"\bam i doing (it|this) (right|correctly)\b",
                r"\bis this (right|correct)\b",
                r"\bcheck( my)? (step|action)\b",
                r"\bam i right\b"
            ]),
            (Intent.REPEAT_STEP, [
                r"\brepeat( the)? (procedure|step|instruction)\b",
                r"\bsay( that)? again\b",
                r"\bwhat was (that|the step)\b",
                r"\brepeat\b"
            ]),
            (Intent.PAUSE_EXPERIMENT, [
                r"\bpause( the)? experiment\b",
                r"\bpause\b",
                r"\bhold on\b"
            ]),
            (Intent.RESUME_EXPERIMENT, [
                r"\bresume( the)? experiment\b",
                r"\bresume\b",
                r"\bcontinue\b"
            ]),
            (Intent.RESET_EXPERIMENT, [
                r"\breset( the)? experiment\b",
                r"\breset\b",
                r"\bstart over\b"
            ]),
            (Intent.STOP_EXPERIMENT, [
                r"\bstop( the)? experiment\b",
                r"\bend( the)? experiment\b",
                r"\bstop\b",
                r"\bterminate\b"
            ]),
            (Intent.SNAPSHOT, [
                r"\b(take|capture)( a)? snapshot\b",
                r"\bsnapshot\b",
                r"\btake photo\b"
            ]),
            (Intent.RECORD_START, [
                r"\bstart recording\b",
                r"\brecord video\b"
            ]),
            (Intent.RECORD_STOP, [
                r"\bstop recording\b"
            ]),
            (Intent.HELP, [
                r"\bhelp\b",
                r"\bcommands\b",
                r"\bwhat can you do\b"
            ])
        ]

    def parse(self, text: str) -> Intent:
        if not text:
            return Intent.UNKNOWN
        cleaned = text.strip().lower()
        # Remove punctuation
        cleaned = re.sub(r"[^\w\s]", "", cleaned)

        for intent, patterns in self.intent_patterns:
            for pattern in patterns:
                if re.search(pattern, cleaned):
                    return intent

        return Intent.UNKNOWN

command_parser = CommandParser()
