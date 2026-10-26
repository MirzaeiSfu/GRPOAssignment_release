# Overview
_If everyone is using an LM, can we manipulate everyone else's with our own?_

We've seen how to use language models for sentiment analysis, which takes in a headline and predicts its sentiment. In this assignment, the language model will instead take a headline and a new, _target_ sentiment. Its goal is to output a headline spun to have the target sentiment while staying similar to the original. To illustrate:
```
headline: "stock market crashes",
target sentiment: "positive"
plausible output: "stock market crashes, buy now!!" 
```
To make the language model effective at this task, we will finetune it using **g**roup **r**elative **p**olicy **o**ptimization (GRPO). The assignment is designed to be more like a group research machine learning project than a single-person course project you may have encountered in previous courses. You should solve this assignment without AI. 

### Group Relative Policy Optimization (GRPO)
GRPO is a technique for finetuning a pretrained language model to yield outputs that maximize a particular, fixed reward function. The high-level idea is to repeat the following process:
1. **Prompting.** We draw a prompt from the data distribution, and use it as a prompt for language model, which returns a group of many completions for the prompt.
2. **Advantage computation.** Another model assigns each completion in the group a reward. From this, each completion is then given an **advantage**, which expresses how high the completion's reward was relative to the group.
3. **Optimization.** The language model is then trained to maximize the probability of the responses, weighted by their advantages. Over time, the language model will hopefully assign higher probabilities to prompt completions that give larger rewards. For additional details, see section four of the [original GRPO paper](https://arxiv.org/pdf/2402.03300).

# What You Need to Do

First, go through the **Setup** instructions, so you have the starter code, can collaborate, and can submit properly. _These look more difficult than they are._ Then, you will implement some parts of the GRPO algorithm that have been removed **(part I)**, document your process for improving GRPO-finetuned models **(part II)**, and achieve improvement on the headline manipulation task **(part III)**.

## Setup
**First**, make sure your **Coursys assignment group** is correct, and create one if you don't have one already! This section describes how to get the starter code, how to use Comet for collaboration and turning in your submissions, and how to run code on GPUs.

### Creating Your Own Code Repo
Each Coursys assignment group will work in a single GitHub repo. This will ease collaboration and be a component of documenting your work.

Creating the repo (steps 2–4) should be done by _one person_ per Coursys assignment group.
1. If you don't already have a GitHub account, get one.
2. Go to the [starter code GitHub](https://github.com/MirzaeiSfu/GRPOAssignment_release).
3. In the top right corner, click *“Use this template”* and then *“Create a new repository”*. Set its name to the Coursys assignment group name. _To avoid publishing your solution online, set sharing to private!_
4. Click _Settings_, then _Collaborators_, and _Add people_. You should be able to add everyone in your Coursys assignment group by the emails used for their GitHub accounts. **(4.1)** Also add the TA emails, `mirzaei@sfu.ca` and `tme3@sfu.ca`.

### Comet MLOps
Using online MLOps tracking is an important aspect of contemporary machine learning, and makes it easy to showcase your work in this assignment.
1. Account creation
    1. Create a Comet account [here](https://www.comet.com/signup/opik?from=llm). You can probably use the academic plan.
    2. Get your API key. Click the person icon in the top-right corner, and then _API Key_. Copy this somewhere safe.
2. Workspace/project setup. _This should be done by one person per group._
    1. **Workspace/project naming.** Use your Coursys assignment group name as the name of the workspace and as the name of the project created in the next step. _If you don't have a Coursys group yet, create one!_
    2. **Creation.** Create a workspace and then a project. Go to your [account settings](https://www.comet.com/account-settings/workspaces) and then click the _Create workspace_ button at right. Once this is done, create it, keeping the visibility private. _**Note:**both the workspace and project are case-insensitive._
    3. **Sharing.** Click the person icon in the top-right corner, then _Invite member_, and add the emails of people you want to let participate in the project. **(3.1)** Also add the TA emails, `mirzaei@sfu.ca` and `tme3@sfu.ca`.
3. **Using the workspace/project.** In your code, edit the defaults for `--comet_project` and `--comet_workspace` to the ones you created. You could also hardcode these in the `comet_ml.start()` function. Commit and push so that everyone running your code will have results go to the same place.

- Finding the views page for the workspace/project can be somewhat nontrivial. It should be found at `https://www.comet.com/WORKSPACE/PROJECT`.
- Run `TrainGRPO.py` with  `--comet online` in your training arguments; it will prompt for your API key.

### Environment Setup
You should be able to run `ImplementMe.py` on any environment where you can install a requisite Python environment.  However, running `TrainGRPO.py` will require an Nvidia GPU. If you have one and are already confident using PyTorch, it will likely be easiest to just work locally. Otherwise, you can get a similar experience by connecting to a Google Colab runtime.

#### Running Locally
Clone the repo and setup up a Python environment as in `README.md`. You should be good to go.

#### Running with a Google Colab runtime
Here, the free GPU access is via a Google Colab runtime. For this you will need a Google account. _Normally, one would run code in Google Colab using the browser interface, which allows you to run code blocks, but isn't convenient for running Python scripts or managing a codebase, which is best done through a command line interface._ To achieve this, either of the following methods will give basically the same interface:
- **VSCode.** You probably have it already installed, if not, go [here](https://code.visualstudio.com/).
- **GitHub Codespace.** This is a VSCode-like interface used in your browser. Go to your GitHub project page, click _<> Code_, and then  `Codespaces`. _**Note:** changes you make are definitely persisted if you commit and push them, but it's not obvious to us just how well they'd persist if a codespace is closed or otherwise exited._ 

**Accessing a Colab runtime.** If using VSCode, clone the project locally. Otherwise, enter a GitHub codespace.
1. In the bottom-left, click the gear icon, choose _Extensions_, type `Colab`, and install the `Colab` extension.
2. Select the empty `ipynb_file.ipynb` file. Towards the top-right, there will be a button to select a kernel for running `.ipynb` files. The button might say `Select kernel` or `Python 3 (ipykernel)`, or maybe something else.
3. Click the button. In the pop-up menu, click _Select another kernel > Colab > New Colab Server > GPU > T4 > Latest_ and hit enter. If there are authentication steps requiring Google sign in, you may have to restart the process. Finally, select `Python 3 (ipykernel)` from the _Select a Kernel from Colab GPU T4_ menu showing that the end.
    - It may be that you already have a session running; if so you can probably use it.
    - _If you have paid pro Colab plan, you can use a faster GPU too!_
4. Open the Command Palette (`ctrl+shift+p` or on macOS `cmd+shift+p`), and type `Colab: Open Terminal`. In the terminal pane (if not showing, click _View > Terminal_), at right you should have two options, `bash` and `Colab GPU T4`. _These are different terminals, each controlling different things, with unconnected filesystems._:
    1. **Bash terminal.** This will let you manipulate the state of the codebase. For example, after you make progress, you can commit and push your changes. Its filesystem matches the one that the files you can edit live on. For working on `ImplementMe.py` in part I, you can run the `pip install ...` commands for the Python environment given in `README.md` here and use this one, since a GPU isn't needed.
    2. **Colab terminal.** This is where you can run code on a GPU. When active, the left terminal subpane should display `/content #` and running `nvidia-smi` in it should display your GPU. To move files/folders to it, right-click them and hit _Upload to Colab_. They should display if you run `ls` in the `/content` directory. _You need to do this each time you modify a file!_. 
5. In the Colab terminal, before running code with GPU support, run the `pip install ...` commands for the Python environment given in `README.md`. Now you should be good to go.


### Starting a Training Run
After completing part I and the environment setup, run training from the repository directory on a GPU. The defaults `--lr 0` and `--eval_iter 0` disable learning and validation respectively, so set a nonzero learning rate and enable validation explicitly.

For example, replace `YOUR_GROUP_NAME` with your Comet workspace and project name:
```bash
python TrainGRPO.py --lr 1e-5 --eval_iter 1 --comet online --comet_workspace YOUR_GROUP_NAME --comet_project YOUR_GROUP_NAME
```
This uses the default 32 training epochs, evaluates after every epoch, and logs to Comet. It is a starting configuration for experimentation, not a guarantee of the reward required for full credit. Tune the existing hyperparameters for parts II and III.

## Part I [10 points]
In the starter code, implement the functions in `ImplementMe.py` that underlie the GRPO algorithm. Implementations must be completely vectorized (no loops!). The points awarded for each are proportional to their difficulty. Specifically:
| Function | Points |
| --- | ---: |
| `compute_grpo_loss` | 1 |
| `output_ids_to_masks` | 3 (1 per returned mask) |
| `get_per_token_logprobs` | 2 |
| `grpo_rl_objective` | 2 |
| `grpo_kl_loss` | 1 |
| `rewards_to_advantages` | 1 |

Implement the six functions listed above. `output_ids_to_masks()` returns the prompt, completion, and attention masks, each worth one point. The supplied `output_ids_to_prompt_mask`, `output_ids_to_completion_mask`, and `output_ids_to_attention_mask` wrappers test these outputs separately; do not implement or modify the wrappers.

Partial credit is proportional within each function or mask: `earned points = allocated points * passed tests / total tests for that function or mask`. Add these contributions to obtain the part I score out of 10. For example, passing 7 of 14 tests for `get_per_token_logprobs` earns `2 * 7 / 14 = 1` point. Each of the three masks is scored separately out of 1 point. Use the exact test counts, not rounded pass percentages, when calculating marks.

The test cases are written so that if your implementation passes them, it is very likely correct, and you should move on to parts II and III.

## Part II [20 points]
With part I finished, you will have a working `TrainGRPO.py` script. It finetunes language models to perform the headline sentiment manipulation task. For both parts II and III, improve performance only by tuning the existing hyperparameters available in `get_args()`. Do not introduce new algorithms or change the provided training algorithm, reward function, or evaluation procedure. Completing the required implementations in part I is still required.

In this portion of the assignment, you will be evaluated on only your documentation of this process. Specifically, you need to document how you create your most performant model, as well as give the documentation for **two** "experiments" performed in the process of getting to it (though you may need/want to run more).

**[16pts total, eight per experiment]** For each experiment, your aim is to better understand the relationship between dependent variables of interest (validation reward, training speed, optimization stability, etc.) and independent variables you control (learning rate, other hyperparameters, etc.), thereby conveying information about how to improve (or not improve) the finetuning process. To make this information high-quality, it's important for the process of achieving it to be well-documented.

Specifically, for each experiment, answer the following in the `Experimentation.md` file:
- **[2pts]** What is the dependent variable (ie. what you would like to influence by changing an independent variable)? If not validation reward, why are they important?
- **[2pts]** What is the independent variable (ie. one you can manipulate), and  what settings of it will you consider? Why would different settings of the independent variable plausibly be related to the dependent variables?
- **[1pts]** Are there any nuisance variables (ie. ones that could introduce noise to results)? If so, are you going to control for them? Why or why not?
- **[1pts]** Documentation—how can you identify **(a)** runs associated to the experiment later, **(b)** each of their precise configurations (a set of variables such that results are deterministic given them—ie. hyperparameters, code, random seed(s), etc.), and **(c)** the results that each achieved?
- **[2pts]** Summarize the results, and draw conclusions relating the independent and dependent variables. Optionally, note down any next steps or useful things you observed.

Finally,

**[4pts]** For the most performant model, document its complete hyperparameter setting and identify the existing hyperparameters you tuned. Algorithmic changes are not permitted in parts II or III.

Experimentation is an inherently messy process, and so it's not necessarily the case that what you write will make what you did seem perfect. Therefore, grading is based on how well you communicate the variables of interest, document how you tested them (eg. command line arguments), document the results and conclusions—not whether the experiment is a success. _Towards the end of this document, there is a sample piece of experiment documentation as an example of what's expected._

## Part III [10 points]
The last part of the assignment is simply a measure of the validation reward achieved by your best model. The grading is simple: your score is `max(0, min(10, 10 * reward_mean_val / 1.8))`. Scores are bounded between 0 and 10 points; negative validation rewards receive 0 points. In other words, if you can get `reward_mean_val` of at least `1.8`, you should expect full credit for this part provided you achieve it only through tuning multiple existing hyperparameters, following the restrictions in part II (no training on the validation set, though you don't need to worry about cross-validation.)

# What To Submit
Submission is on Coursys. You will **(1)** submit your finished `ImplementMe.py`, **(2)** submit your writeup for part II as an `Experimentation.md` file, **(3)** submit your code as a GitHub link (remember to share the repo!) with the git tag used for your final run, **(4)** Submit a link to your Comet workspace/project (remember to share it!), and **(5)** submit a link to your best run on Comet

# Hints
- You can complete part I without a GPU
- You don't need more than 32 epochs—on a Google Colab T4, your runs probably should probably take 30–50 minutes (and it's valid to do hyperparameter tuning in "smaller" settings that take less time).

# Experimentation Example
- **Dependent variables.** The dependent variable will be the validation reward, as it's the variable of interest. By optimizing for this, we may overfit it in hyperparameter tuning, but not in training an individual model.

- **Independent variables.** We consider the hyperparameter `--grpo_kl_weight`, which controls how much the trained policy is incentivized to stay close to the reference policy while also trying to maximize reward. Intuitively, if set too high, it would be difficult for the trained policy to adapt to yield better completions. If set too low, it could focus too much on maximizing a reward function, thereby removing desireable traits learned in prior training. We will try values in `0.01, 0.02, 0.04, 0.08, ...`.

- **Nuisance variables.** Important nuisance variables include the learning rate, and seeding. In order to conserve compute, we leave whether results we attain are robust to different random seeds an open question. Hypothetically, the scale of the KL loss and the rewards for the KL objective could be very different, and so changing the weight of the KL loss could vary gradient magnitudes. This could make it important to adapt the learning rate. Therefore, we will consider roughly logarithmically spaced values of `lr=..., 1e-5,3e-5, 1e-4, 3e-4, ...`, tuning the learning rate independently for different settings of `grpo_kl_weight`, and compare them using results attained with their best learning rate.

- **Documentation.** The runs, including their configurations, results, and code, are logged online in Comet with tag `Oct23-BatchSizeTest-0-1`.

- **Results.** Since here learning rate is treated as a nuisance variable, we compare different settings of `grpo_kl_weight` run in their optimal learning rate. We find that setting `grpo_kl_weight=some_value` seems to be broadly more performant in attaining superior validation reward, with optimal learning rate `lr=another_value`.
