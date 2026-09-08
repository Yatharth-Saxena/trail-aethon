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
    GREETING = "GREETING"
    STATUS = "STATUS"
    MISSION_DETAILS = "MISSION_DETAILS"
    PROCEDURE_OVERVIEW = "PROCEDURE_OVERVIEW"
    PROGRESS = "PROGRESS"
    SAFETY_CHECK = "SAFETY_CHECK"
    WHICH_HAND = "WHICH_HAND"
    SPEED_CHECK = "SPEED_CHECK"
    FUN_FACT = "FUN_FACT"
    MUTE = "MUTE"
    UNKNOWN = "UNKNOWN"

# Comprehensive wake-word patterns covering all phonetics and speech-to-text transcriptions
WAKE_TWO_WORD_VARIANTS = (
    r"ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|"
    r"eden|aeon|item|anton|titan|python|than|then|turn|tane|thanks?|ton|tan|aton|chetan"
)

WAKE_SINGLE_WORD_VARIANTS = (
    r"ae?thon|ae?than|ae?thane|athan|athlon|ethan|ethane|eaton|athena|atom|aidan|"
    r"eden|aeon|item|anton|titan|python|than|then|turn|tane|aton|hetan|chetan"
)

WAKE_WORD_PATTERNS = [
    # Repeated wake calls e.g. 'ethane than', 'ethan ethan', 'aethon aethon', 'he tane tane tan'
    rf"^\s*(?:ethane|ethan|ae?thon|ae?than|athan|tane)\s+(?:than|then|ethan|ethane|ae?thon|ae?than|athan|tane|tan)\b[,.\s]*",
    # Two-word call: Hey / Hi / Hello / OK / Okay / He / A / Suno + name variant
    rf"^\s*(?:hey|hi|hello|ok|okay|he|a|ey|ay|suno)\s+(?:{WAKE_TWO_WORD_VARIANTS})\b[,.\s]*",
    # Anywhere in phrase two-word call (e.g. "uh hey aethon", "please hey aethon")
    rf"\b(?:hey|hi|hello|ok|okay|he|suno)\s+(?:{WAKE_TWO_WORD_VARIANTS})\b",
    # Single word fast call (hetan, ethan, ethane, aethon, aethan, etc. - NOT standalone 'thanks')
    rf"^\s*(?:{WAKE_SINGLE_WORD_VARIANTS})\b[,.\s]*",
]

def has_wake_word(text: str) -> bool:
    if not text:
        return False
    for pat in WAKE_WORD_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# Prefix patterns applied in order when removing a wake phrase. Ordered
# longest-form first so "ethane than" is consumed as one call rather than
# leaving a stray "than" behind.
WAKE_STRIP_PATTERNS = (
    rf"^\s*(?:ethane|ethan|ae?thon|ae?than|athan|tane)\s+(?:than|then|ethan|ethane|ae?thon|ae?than|athan|tane|tan)\b[,.:;!?\s]*",
    rf"^\s*(?:hey|hi|hello|ok|okay|he|a|ey|ay|suno)\s+(?:{WAKE_TWO_WORD_VARIANTS})\b[,.:;!?\s]*",
    rf"^\s*(?:{WAKE_SINGLE_WORD_VARIANTS})\b[,.:;!?\s]*",
)


def strip_wake_word(text: str) -> str:
    """
    Remove a leading wake phrase and return just the command.

    Public counterpart to the frontend's `stripWakeWord()`, so the browser's
    wake-word gating can be validated server-side rather than trusted.
    Returns an empty string when the utterance was only a wake call.
    """
    if not text:
        return ""
    cleaned = text.strip()
    for pat in WAKE_STRIP_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def split_wake_command(text: str) -> tuple[bool, str]:
    """
    Return `(had_wake_word, command_text)` for an utterance.

    Mirrors the frontend contract: the caller can reject anything without a
    wake word outright, and act on the remaining command text.
    """
    return has_wake_word(text), strip_wake_word(text)

class CommandParser:
    def __init__(self):
        self.intent_patterns = [
            (Intent.START_EXPERIMENT, [
                r"\bstart( the)? experiment\b",
                r"\bbegin( the)? experiment\b",
                r"\blaunch experiment\b",
                r"\bstart( the)? (test|procedure)\b",
                r"^\s*start\s*$",
                r"^\s*begin\s*$",
                r"^\s*launch\s*$"
            ]),
            (Intent.STATUS, [
                r"\bsystem status\b",
                r"\bstatus report\b",
                r"\bgive me (a )?status\b",
                r"\bhow is (everything|the system|aethon)\b",
                r"\ball systems\b"
            ]),
            (Intent.WHAT_AM_I_DOING, [
                r"\bwhat( am i| i am| is being| is) doing\b",
                r"\bwhat is happening\b",
                r"\bwhat is going on\b",
                r"\bwhat('?s| is) my action\b",
                r"\bcurrent action\b",
                r"\bdescribe( my)? action\b",
                r"\bwhat action is (going on|detected)\b",
                r"\bwhat (am i|i am) doing right now\b",
                r"\bwhat (am i|i am) doing\b"
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
                r"\bwhat objects? do you see\b",
                r"\bwhat is this\b",
                r"\bwhat is that\b",
                r"\bwhat do you see\b"
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
                r"\bwhat are my (movements?|moments?)\b",
                r"\bwhat (movement|moment)\b",
                r"\btrack (movement|movements|moment)\b",
                r"\bhow am i moving\b",
                r"\bdetect (movement|moment)\b",
                r"\bposture and (movement|movements)\b",
                r"\bwhat is my posture\b",
                r"\bmovement analysis\b"
            ]),
            (Intent.NEXT_STEP, [
                r"\b(what('?s| is) )?(the )?next step\b",
                r"\btell me (the )?next step\b",
                r"\bwhat do i do( next)?\b",
                r"^\s*(what('?s| is) )?(the )?procedure\s*$",
                r"\bnext procedure\b",
                r"\bwhat to do\b",
                r"\bwhat next\b",
                r"^\s*next\s*$",
                r"\bguidance\b",
                r"\bwhat step\b",
                r"\bwhat should i do\b"
            ]),
            (Intent.CHECK_CURRENT_ACTION, [
                r"\bam i doing (it|this) (right|correctly)\b",
                r"\bis this (right|correct)\b",
                r"\bcheck( my)? (step|action)\b",
                r"\bam i right\b",
                r"\bis this correct\b"
            ]),
            (Intent.REPEAT_STEP, [
                r"\brepeat( the)? (procedure|step|instruction)\b",
                r"\bsay( that)? again\b",
                r"\bwhat was (that|the step)\b",
                r"\brepeat\b",
                r"\bcan you repeat\b"
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
                r"\bwhat can you do\b",
                r"\bwhat commands\b"
            ]),
            (Intent.MISSION_DETAILS, [
                r"\b(what('?s| is) )?(the )?mission\b",
                r"\bexplain( the)? mission\b",
                r"\babout( this)? experiment\b",
                r"\bwhat are we doing\b",
                r"\bwhat is payload assembly\b",
                r"\bmission (overview|goal|objective|details)\b",
                r"\bwhat is the goal\b",
                r"\bwhy are we doing this\b",
                r"\bmission brief\b"
            ]),
            (Intent.PROGRESS, [
                r"\bhow many steps (left|remaining|done|completed)\b",
                r"\b(what is )?(my |the )?progress\b",
                r"\bhow far (am i|are we)\b",
                r"\bprogress percentage\b",
                r"\bpercentage( complete)?\b",
                r"\bcurrent progress\b"
            ]),
            (Intent.PROCEDURE_OVERVIEW, [
                r"\b(show|tell|list|what are) (all )?(the )?steps\b",
                r"\bprocedure overview\b",
                r"\bhow many steps\b",
                r"\bwhat is the sequence\b",
                r"\blist steps\b",
                r"\ball steps\b",
                r"\bentire procedure\b",
                r"\bsequence of steps\b"
            ]),
            (Intent.SAFETY_CHECK, [
                r"\bsafety( check| rules| guidelines| protocol| precautions)?\b",
                r"\bis it safe\b",
                r"\bprecaution(s)?\b",
                r"\bhazard(s)?\b",
                r"\bsafety status\b"
            ]),
            (Intent.WHICH_HAND, [
                r"\bwhich hand\b",
                r"\bwhat hand\b",
                r"\bam i using (my )?(left|right) hand\b",
                r"\bhand tracking\b",
                r"\bdetect hand\b"
            ]),
            (Intent.SPEED_CHECK, [
                r"\bhow fast\b",
                r"\bcheck( my)? speed\b",
                r"\bam i moving too fast\b",
                r"\bmovement (speed|velocity|rate)\b",
                r"\bvelocity check\b"
            ]),
            (Intent.FUN_FACT, [
                r"\b(space |fun )?fact\b",
                r"\btell me (a |something )?(space |fun )?fact\b",
                r"\btrivia\b",
                r"\btell me a joke\b",
                r"\bspace trivia\b"
            ]),
            (Intent.MUTE, [
                r"\b(chup|chup raho|shut up|be quiet|silence|mute|stop talking|stop speaking)\b"
            ]),
            (Intent.GREETING, [
                r"\b(hello|hi|hey|howdy|greetings|good morning|good afternoon|good evening|good day)\b",
                r"\b(namaste|namaskar|kaise ho|kya haal|kya chal raha)\b",
                r"\bhow are you( doing)?\b",
                r"\bhow('?s| is) it going\b",
                r"\bhow are things\b",
                r"\bwhat'?s up\b",
                r"\bwho are you\b",
                r"\bwhat is your name\b",
                r"\bwhat('?s| is) your name\b",
                r"\bintroduce yourself\b",
                r"\btell me about yourself\b",
                r"\bwho (made|created|developed) (you|aethon)\b",
                r"\bthank(s| you)( so much)?\b",
                r"\bappreciate it\b",
                r"\b(good|great) job\b",
                r"\bare you (there|listening|online|ready)\b",
                r"\bcan you hear me\b",
                r"\b(bye|goodbye|see you|good night)\b"
            ])
        ]

    def _strip_wake_word(self, text: str) -> str:
        """Remove wake-word prefix from the text so intent matching works
        on the actual command part."""
        return strip_wake_word(text)

    def parse(self, text: str) -> Intent:
        if not text:
            return Intent.UNKNOWN
        cleaned = text.strip().lower()
        # Remove punctuation
        cleaned = re.sub(r"[^\w\s']", "", cleaned)
        # Strip wake words
        cleaned = self._strip_wake_word(cleaned)

        if not cleaned:
            # Only the wake word was said → treat as greeting
            return Intent.GREETING

        for intent, patterns in self.intent_patterns:
            for pattern in patterns:
                if re.search(pattern, cleaned):
                    return intent

        return Intent.UNKNOWN

command_parser = CommandParser()
