# Self-paced ARENA 3.0

Basic strategy is to always use rebase to keep your local main branch up to date with upstream main.

## How to set up your local repository:

1. Fork the repo on GitHub first (do this in browser)
    > Go to https://github.com/callummcdougall/ARENA_3.0 and click "Fork"

2. Make original repo as "upstream"
    > git remote add upstream https://github.com/callummcdougall/ARENA_3.0.git

3. Add your fork as "origin"
    > git remote add origin git@github.com:YOUR_USERNAME/ARENA_3.0.git

4. Push your main branch to your fork
    > git push -u origin main


## Daily Workflow

1. Fetch updates from upstream
    > git fetch upstream

2. Rebase updates into your local main branch
    > git checkout main
    > git rebase upstream/main

3. Push updates to your fork
    > git push origin main

## How to do solutions

In my case, my source solutions branch is `option2-python-vscode/solutions`.

Every other solution branch will branch out from this.

1. Create a new branch for the solution
    > git checkout -b option2-python-vscode/solutions/chapter_0/part0_prereqs

2. Make your changes
    >  git commit -m "Add solution for chapter 0, part 0, prereqs"

3. Push your changes to your fork
    > git push origin option2-python-vscode/solutions/chapter_0/part0_prereqs

## How to do own tooling

An example might be you prefer to use `uv` over `pip`

1. Create a new branch for the tooling from main
    > git checkout -b experiments/uv

2. Make your changes
    > git commit -m "Add uv as an alternative to pip"

3. Push your changes to your fork
    > git push origin experiments/uv


## Pdf of streamlit

Instead of opening the streamlit locally, I find it easier to just go to the hosted website at https://arena-chapter0-fundamentals.streamlit.app/ and keep a pdf copy in /pdf-of-streamlit

## Option chosen: Python file and VS Code

To access the course, there are 3 options given at https://arena-chapter0-fundamentals.streamlit.app/#how-to-access-the-course

I choose option 2: Python file and VS Code (also the strong recommendation for all in-person participants).

### Workflow for Each Part

1. **Open the Streamlit app** for the chapter you're working on:
   - Chapter 0: https://arena-chapter0-fundamentals.streamlit.app/

2. **Navigate to the exercises** by clicking on the part in the sidebar (e.g., "0️⃣ Prerequisites")
   - The Streamlit page contains explanations, examples, and exercise descriptions
   - Each exercise section has a Colab link for reference (but we won't use Colab)

3. **Create a `solutions.py` file** in the corresponding exercises folder:
   ```
   chapter0_fundamentals/exercises/part0_prereqs/solutions.py  # yours
   ```

4. **Import the utilities at the top of your `solutions.py`**:
   ```python
   import sys
   from pathlib import Path

   # Ensure the exercises directory is in the path
   section_dir = Path(__file__).resolve().parent
   exercises_dir = section_dir.parent
   if str(exercises_dir) not in sys.path:
       sys.path.append(str(exercises_dir))

   # Now import from utils and tests
   from part0_prereqs import tests
   from part0_prereqs.utils import arr, display_array_as_img, display_soln_array_as_img
   ```

5. **Work through each exercise**:
   - Read the exercise description in Streamlit
   - Write your solution in `solutions.py`
   - Use `# %%` cell markers for VS Code's interactive Python features
   - Run tests to verify: `tests.test_einsum_trace(your_function)`

6. **Run your code** using VS Code:
   - `Shift+Enter` to run the current selection/line
   - Click "Run Cell" above any `# %%` marker

## Pacing

Given I am juggling work, and a part-time masters degree in Singapore Management University, taking this arena course is more of a upskilling exercise than a full course.

My priority rule is, "Revenue > School > Upskilling".

Given this, I will proceed with doing Arena 3.0 using the Concept Sprints.

* *Status:* **"Concept Sprints" (Long Weekends Only).**
* *Strategy:* Break Chapter 0 into atomic "Done" states to kill the [Zeigarnik Effect](https://www.psychologytoday.com/sg/basics/zeigarnik-effect#:~:text=The%20Zeigarnik%20Effect%20is%20the,easily%20recalled%20than%20completed%20tasks.).
* *The Sprints:*
    * **Sprint 1:** `part1_raytracing` (Stand-alone). *Goal: All cells in .ipynb run.*
    * **Sprint 2:** `part2_cnns` (Depends on Sprint 1). *Tactic: Allocate first 30 mins for "Sprint 1 Refresher".*
    * **Sprint 3:** `part3_optimization` (Depends on Sprint 2).
* *Rule:* **Never open the next folder until the next sprint.**
* *Constraint:* Do not touch this on weeknights. Only during Breaks/Long Weekends.