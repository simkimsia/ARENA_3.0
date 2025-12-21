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

## Option chosen: Python file and vscode

To access the course, there are 3 options given at https://arena-chapter0-fundamentals.streamlit.app/#how-to-access-the-course

I choose option 2: Python file and vscode

Which is also the strong recommendation for all in-person participants.

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