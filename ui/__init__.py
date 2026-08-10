# Leaf modules (state/panels/sidebar/theme) must never import from
# ui_pages, and must not import each other except ingestion -> state
# and chat -> panels. app_ui.py references ui_pages/*.py only as path
# strings passed to st.Page(), never as Python imports - keep it that
# way so the dependency graph stays one-directional:
#   ui_pages/* -> ui/* -> cli.py, app.*
