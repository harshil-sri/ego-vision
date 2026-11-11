"""Names shared by the data, training and app code."""

#: Extra class of verb heads trained on sliding windows: "none of the demo
#: verbs is happening" (another verb, or no action at all). The engine
#: never fires it; see CookingEngine._recognize_verb.
#: Label ids are ALPHABETICAL over all class names (the smoothers sort their
#: verb list), so "none" sits between "mix" and "open", not at the end.
NONE_VERB = "none"
