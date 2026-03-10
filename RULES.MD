## 🌍 GLOBAL RULES FOR ALL PROJECTS

### PROJECT STRUCTURE ENFORCEMENT

The root directory shall contain ONLY these specific file types: README.md for project overview and quick start, requirements.txt or pyproject.toml for dependencies, package.json if it's a Node.js project, docker-compose.yml for container orchestration, Dockerfile for container definition, .gitignore for version control exclusions, .env.example as environment variables template, LICENSE for legal stuff, and mkdocs.yml if you're building a documentation site.

NO OTHER FILES ARE ALLOWED AT ROOT LEVEL. All code, documentation, and assets must be placed in appropriate subdirectories. This keeps the project root clean and immediately understandable to anyone who clones the repository.

### README STRUCTURE REQUIREMENTS

The README must open with a clear project title and a one-paragraph executive summary explaining what the project does, why it exists, and who it's for. This summary must be understandable by someone with zero context about the project.

Immediately following must be a badges section showing build status, test coverage, license type, latest version, and any other relevant status indicators. These badges give instant visual feedback about project health.

### README UPDATE PROTOCOL

The README must be updated whenever ANY of the following occur. When new directories or files are added to the project, the structure section must be updated immediately. When dependencies change, the technology stack and setup instructions must be updated. When configuration variables change, the configuration guide must be updated. When APIs change, the API documentation must be updated. When deployment procedures change, the deployment guide must be updated. When any error message or common issue is identified, the troubleshooting guide must be updated. When new features are added, the architecture explanation and roadmap must be updated. When old features are deprecated, all relevant sections must be updated and deprecation notices added.

The README must have a last updated timestamp at the top showing when significant changes were made. A changelog section at the bottom can summarize major updates to the README itself.

### README QUALITY REQUIREMENTS

The README must be written in clear, professional English accessible to both technical and non-technical readers. Code blocks must be properly formatted with language specified. Commands must be clearly distinguished from output. Examples must be complete and tested. Links must be verified and updated. Screenshots and diagrams must be included where they aid understanding.

No question about the project should remain unanswered after reading the README. Anyone cloning the repository should have everything they need to understand, set up, use, contribute to, deploy, and maintain the project without asking for help.

The README is the front door to your project on GitHub. It must be welcoming, comprehensive, accurate, and maintained with the same rigor as your source code. A complete README signals professionalism, saves countless support hours, and attracts quality contributors.

This rule works in perfect harmony with your project_structure.md rule - the structure file maintains the detailed tree for internal reference, while the README presents it beautifully for the world to see along with everything else anyone could possibly need to know.

### VIRTUAL ENVIRONMENT AUTOMATION

For EVERY new project, you must create a SETUP_ENV directory that contains platform-specific setup scripts.

For Windows, create setup_venv.bat that handles the complete environment setup. The script should first display a message showing step 1 of 4 creating virtual environment using python -m venv venv. Then step 2 should show activating virtual environment by calling venv\Scripts\activate.bat. Step 3 upgrades pip using python -m pip install --upgrade pip. Step 4 checks if requirements.txt exists and if so installs all requirements with pip install -r requirements.txt showing success message, otherwise shows a notice that no requirements.txt was found. Finally display setup complete message and indicate that a log file has been saved with timestamp in the filename showing date and time. End with pause so user can see the results.

For Linux and Mac, create setup_venv.sh that does the same but with bash syntax. Show step 1 of 4 creating virtual environment using python3 -m venv venv. Step 2 activates with source venv/bin/activate. Step 3 upgrades pip using pip install --upgrade pip. Step 4 checks for requirements.txt and installs if present. Mark completion with timestamped log file. Make sure to mention that users need to run chmod +x setup_venv.sh to make it executable before running.

Also include a README_SETUP.md file in the SETUP_ENV directory that gives clear instructions. For Windows users tell them to navigate to SETUP_ENV directory and run setup_venv.bat. For Linux and Mac users tell them to navigate to SETUP_ENV directory, run chmod +x setup_venv.sh to make it executable, then run ./setup_venv.sh. Emphasize that all installation logs will be visible in real-time during setup and also saved to a log file for later debugging if needed.

## 📁 PROJECT-SPECIFIC RULES

### RULE 1: PROJECT STRUCTURE DOCUMENTATION

A file named project_structure.md MUST be maintained at the project root with complete tree representation including descriptions for everything.

The file should start with a title showing Project Structure for whatever the project name is and include last updated date stamp.

Then show the root level tree with proper indentation showing the project root folder, then list README.md with description that it contains project overview setup and usage, requirements.txt with description that it lists python dependencies with versions, Dockerfile with description that it defines container configuration, docker-compose.yml with description that it handles multi-container setup, .gitignore with description that it lists git exclusion rules, .env.example with description that it provides environment variables template, LICENSE with description showing what license type like MIT or Apache or GPL, and finally project_structure.md with description that this is the file itself containing the project map.

Then document the core directories section. Show src folder as source code with proper indentation showing init.py as package initializer, main.py as entry point number one. Then show core subdirectory with init.py, engine.py as main processing engine, and models.py as data models. Then show utils subdirectory with init.py, logger.py for logging configuration, and validators.py for input validation. Then show api subdirectory with init.py, routes.py for route definitions, and middleware.py for request response processing.

Next document the tests directory showing init.py, test_main.py for testing main.py, test_engine.py for engine tests, and fixtures subdirectory with sample_data.json for test data.

Then document the docs directory showing api subdirectory with endpoints.md, guides subdirectory with setup.md, and architecture subdirectory with overview.md.

Then document the progress_logs directory showing progress_1 folder containing 01_planning.md, 02_tasks.md, and 03_report.md, and progress_2 folder with same structure.

Finally document the SETUP_ENV directory showing setup_venv.bat for windows, setup_venv.sh for linux mac, and README_SETUP.md for setup instructions.

The update protocol requires that you update this file whenever directory structure changes. Include brief description for every file and folder. Mark deleted paths with strikethrough formatting to show they're gone. Highlight new paths with bold formatting to draw attention to additions.

### RULE 2: SEQUENTIAL FILE NAMING PROTOCOL

This rule applies to ALL subdirectories at every level but EXCLUDES the root directory itself.

The naming convention requires that files of the same type within any given directory must follow a pattern of sequence number followed by underscore followed by descriptive name followed by dot and extension.

Sequence rules state that you start at 01 for the first file of its type in that specific folder. You increment by 1 for each new file of the same type. You reset the counter for each different file extension you encounter. The sequence numbers persist even if files get deleted meaning you never reuse numbers.

For example in src/utils directory you would have 01_logger.py as the first python file created, then 02_validators.py as the second python file, then 03_helpers.py as the third python file. Then if you add a json file it becomes 01_config.json because it's a different file type so counter resets, followed by 02_settings.json as the second json file.

In docs/guides directory you would have 01_setup.md as first markdown file, 02_deployment.md as second, 03_troubleshooting.md as third. Then first image added becomes 01_diagram.png with counter resetting for the png extension.

In progress_logs/progress_1 directory you would have 01_planning.md as first markdown, 02_tasks.md as second, 03_report.md as third. Then any images like metrics would be 01_metrics.png and graphs would be 02_graph.png.

Enforcement requires always checking existing files before naming new ones. Find the highest existing sequence number for that specific extension in that folder. Add plus one for the new file. Always zero-pad to two digits meaning 01 through 99.

### RULE 3: PROGRESS TRACKING SYSTEM

The progress logs architecture follows a specific directory structure. At the top level you have a progress_logs folder. Inside that you create numbered session folders like progress_1 for the first development session, progress_2 for the second session, progress_3 for the third session and so on continuing this pattern.

Inside each session folder you must have specific files. 01_planning.md contains initial planning and thinking. 02_tasks.md contains task breakdown and execution tracking. 03_report.md contains the complete session report with analysis. You may also include images like 01_metrics.png showing performance metrics if any were generated, and 02_comparison.png showing before and after visuals if relevant.

For the 01_planning.md file it should contain a planning session header indicating which progress number this is. Then a context section describing current state of project, what triggered this session, and any dependencies or blockers. Then a goals section listing primary objective, secondary objective, and nice to have items. Then an approach section explaining strategy one with explanation, strategy two with explanation, and why this particular approach was chosen. Then expected outcomes describing what success looks like and what metrics to track. Then risks section listing potential issues and mitigation plans for each.

For the 02_tasks.md file it should contain a task execution header indicating which progress number this is. Then a task list with clear status indicators. In progress items should show as not checked with description and any subtasks. Completed items should show as checked with description, what output was produced, time duration if tracked, and any observations noted. Blocked items should show as not checked with description, what blockers are preventing progress, and what help is needed to resolve. Then a decisions made section documenting each decision with what was decided and why that choice was made.

For the 03_report.md file it should contain a session report header indicating which progress number this is. Then an executive summary giving brief overview of what happened during this session. Then an actual versus expected table comparing each goal showing what was expected, what actually happened, and gap analysis explaining any differences. Then a detailed work log with three subsections. What we did showing step by step actions. How we did it showing methods and tools used. Why we did it that way explaining reasoning and trade-offs considered. Then results section covering quantitative results like numbers metrics and stats, and qualitative results like observations and insights. Then visual evidence section with figure references and explanations for any images included. Then lessons learned listing key takeaways from the session. Finally next steps identifying immediate next action and future considerations.

For naming within each session folder all files must follow sequential naming. Text files should be 01* something dot md, 02* something dot md, 03* something dot md. Images should be 01* something dot png, 02* something dot jpg. Data files should be 01* something dot json, 02\_ something dot csv.

The update cadence requires creating a new session folder for each and every work session. Update files progressively during the actual work. Complete the 03_report.md file before ending the session.

## 🎯 QUICK REFERENCE CARD

For root directory the allowed files are README.md, requirements.txt, Dockerfile, docker-compose.yml, .gitignore, .env.example, LICENSE, mkdocs.yml, and package.json. Everything else is blocked.

For file naming in subdirectories only, the first python file becomes 01_something.py, second python file becomes 02_something.py, first markdown file becomes 01_something.md with counter resetting per file type.

For progress structure you have progress_logs at top level, inside that progress_1 folder containing 01_planning.md, 02_tasks.md, and 03_report.md, then progress_2 folder with same structure, and so on.

For environment setup you have SETUP_ENV folder containing setup_venv.bat for windows, setup_venv.sh for linux and mac, and README_SETUP.md with instructions.

---
