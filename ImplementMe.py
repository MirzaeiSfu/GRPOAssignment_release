import argparse
import copy
import einops
from functools import partial
import json
import os
import os.path as osp
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback

import UtilsBase
from UtilsBase import to_json_encoding, from_json_encoding  # Use to encode/decode tensors in test cases
from Utils import twrite                                    # Useful for debugging

############ This *should* work automatically, but if not you can configure it #######
def get_terminal_columns(default=200):
    """Returns the number of columns in the terminal, or [default] if this does not
    work properly. Useful for debugging.
    """
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = default
    return columns
torch.set_printoptions(profile="full")
torch.set_printoptions(linewidth=get_terminal_columns())
######################################################################################
######################################################################################
######################################################################################

def compute_grpo_loss(*, rl_loss, kl_loss, grpo_kl_weight=1, **kwargs):
    """Returns the per-token GRPO loss, which is Equation (3) in the GRPO
    paper (arxiv.org/pdf/2402.03300).

    HINT: the paper writes the objective with an eye towards maximizing it, but we
    are going to minimize. Flip signs accordingly.
    
    Args:
    rl_loss         -- (BS C)xL tensor of per-token RL losses. Note that this is a bit
                        of a misnomer, since larger values of this signify that the
                        model is doing better.
    kl_loss         -- (BS C)xL tensor of per-token KL losses
    grpo_kl_weight  -- weight for the KL loss
    **kwargs        -- unused
    """
    raise NotImplementedError()

def output_ids_to_masks(*, output_ids, prompt_input_ids, pad_token_id, eos_token_id, **kwargs):
    """Returns a Namespace with 'attention_mask', 'completion_mask', and
    'prompt_mask' attributes. These respectively map to (BS C)xT 0-1 mask tensors 
    [attention_mask], [completion_mask], and [prompt_mask], defined as follows.
    All three masks must have dtype torch.int32 and contain only 0 and 1.

    Define the PROMPT REGION of each sequence in [prompt_input_ids] as the span of
    tokens from the first non-pad token to the end.

    Define the PROMPT REGION of each sequence in [output_ids] as the span of tokens
    that starts at the first non-pad token in the sequence and matches the prompt
    region of the corresponding sequence in [prompt_input_ids].

    Define the COMPLETION REGION of each sequence in [output_ids] as the span of
    tokens that starts immediately after the prompt region and ends with the first EOS
    token in the sequence after the prompt region ends, or, if there is no EOS token,
    at the end of the sequence.

    1. [attention_mask] is a (BS C)xT tensor whose ij value is 1 if in the ith
        sequence of [output_ids], the jth token is part of the prompt region OR the
        completion region, and 0 otherwise.
    2. In [prompt_mask], the ij index is 1 if the ij token of [output_ids] is part of
        the prompt region of the ith sequence, and 0 otherwise.
    3. In [completion_mask], the ij index is 1 if the ij token of [output_ids] is part
        of the completion region of the ith sequence, and 0 otherwise.

    HINTS:
    1. We consider a particular implementation where we don't assume the padding at
        the start of sequences in [output_ids] matches the padding at the start of
        corresponding sequences in [prompt_input_ids].
    2. Use torch.argmax() to find the start index of each prompt region in a tensor.
        pytorch.org/docs/stable/generated/torch.argmax.html. Infer the number of
        completions per prompt from the shapes of [output_ids] and [prompt_input_ids].
    ----------------------------------------------------------------------------------
    
    Args:
    output_ids          -- (BS C)xT tensor of token IDs returned by applying
                            model.generate() to [prompt_input_ids] together with other
                            arguments.

                            The first C sequences along the batch dimension contain
                            the C completions for the first prompt (first row of
                            [prompt_input_ids]), the next C sequences contain the C
                            completions for the second prompt, and so on.

    prompt_input_ids    -- BSxS tensor of token IDs used as input to model.generate()
    pad_token_id        -- token ID used for padding tokens
    eos_token_id        -- token ID used for the EOS token
    **kwargs            -- unused
    """
    raise NotImplementedError()


######################################################################################
# DO NOT IMPLEMENT THESE! Instead, implement output_ids_to_masks() instead.
# Correct implementations for them a bunch of code, and they all have the same test
# cases. It might be easier to debug each component output separately though, so
# separate functions are provided.
######################################################################################
def output_ids_to_prompt_mask(*, output_ids, prompt_input_ids, pad_token_id, eos_token_id, **kwargs):
    """Returns the prompt mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **kwargs)).get("prompt_mask", None)
def output_ids_to_completion_mask(*, output_ids, prompt_input_ids, pad_token_id, eos_token_id, **kwargs):
    """Returns the completion mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **kwargs)).get("completion_mask", None)
def output_ids_to_attention_mask(*, output_ids, prompt_input_ids, pad_token_id, eos_token_id, **kwargs):
    """Returns the attention mask as defined in output_ids_to_masks()."""
    return vars(output_ids_to_masks(output_ids=output_ids,
        prompt_input_ids=prompt_input_ids,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        **kwargs)).get("attention_mask", None)
######################################################################################
######################################################################################
######################################################################################

def get_per_token_logprobs(*, logits, token_ids, **kwargs):
    """Returns a BSx(T-1) tensor of log-probabilities, whose ij element is the
    log-probability of the j+1th token in the ith sequence of [token_ids], given all
    the preceeding tokens 0...j (inclusive) in the ith sequence of [token_ids].

    HINTS:
    1. Understanding logits: singlestore.com/blog/a-guide-to-softmax-activation-function
    2. torch.gather() is your friend: pytorch.org/docs/stable/generated/torch.gather.html
    3. Prefer log-softmax() over log(softmax()) for numerical stability

    Args:
    logits      -- BSxTxV tensor of logits output by an LM with vocab size V given
                    [token_ids] as input
    token_ids   -- BSxT tensor of token IDs
    **kwargs    -- unused
    """
    raise NotImplementedError()

def grpo_rl_objective(*, logprobs_model, logprobs_pi_old, advantages, grpo_clip=0.2, **kwargs):
    """Returns a BSxL tensor where the ij element is the GRPO policy objective.

    HINT: This is (equation 3) in the original GRPO paper (arxiv.org/pdf/2402.03300),
    but with two changes (1) the KL term is not computed here, and (2) it is defined
    over batches of log-probabilities and advantages that don't necessarily correspond
    to a single group (ie. those from one prompt) of model outputs.

    Args:
    Suppose that originally there was a BSx(L+1) tensor of tokens fed to LMs pi_theta
    and pi_old. The ith sequences in [logprobs_model] and [logprobs_pi_old] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...L in the ith
    sequence of this original tensor. The ith row of [advantages] contains the
    advantage for that completion, repeated across its L token positions.

    logprobs_model  -- BSxL tensor of log-probabilities from the model pi_theta
    logprobs_pi_old -- BSxL tensor of log-probabilities from the old policy pi_old
    advantages      -- BSxL tensor of advantages, matching the log-probability tensors
    grpo_clip       -- clipping parameter for GRPO
    **kwargs        -- unused
    """
    raise NotImplementedError()

def grpo_kl_loss(*, logprobs_model, logprobs_pi_ref, **kwargs):
    """Returns a BSxT tensor whose ij element is the modified KL loss for jth index of
    the ith element in the batch.
    
    Hint: equation (4) in the GRPO paper (arxiv.org/pdf/2402.03300.pdf).

    Args:
    Suppose that originally there was a BSx(T+1) tensor of tokens fed to LMs pi_theta
    and pi_ref. The ith sequences in [logprobs_model] and [logprobs_pi_ref] contain
    conditional log-probabilities for the (zero-indexed) tokens 1...T in the ith
    sequence of this original tensor.

    logprobs_model  -- BSxT tensor of log-probabilities from the model pi_theta
    logprobs_pi_ref -- BSxT tensor of log-probabilities from the model pi_ref
    **kwargs        -- unused
    """
    raise NotImplementedError()
 
def rewards_to_advantages(*, rewards, grpo_completions, **kwargs):
    """Returns a (BS C)-length tensor of advantages computed from [rewards] using
    GRPO's method.

    HINT: Compute the mean and standard deviation within each prompt's group of
    completions. Use the Bessel correction for the standard deviation. Subtract
    the group mean from each reward, then divide by (group standard deviation +
    1e-4) for numerical stability.

    Args:
    rewards             --(BS C)-length tensor of rewards
    grpo_completions    -- C, the number of completions per prompt. At least two.
    """
    raise NotImplementedError()

@torch.no_grad()
def test_fn(*, fn, fn_args, fn_kwargs, fn_expected_output, device="cpu", verbose=False):
    """Tests [fn] with the given arguments and returns the output."""
    try:
        output = fn(*fn_args, device=device, **fn_kwargs)
    except Exception as e:
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function raised exception during evaluation of test inputs. Traceback:")
        tqdm.write(error_trace)
        return argparse.Namespace(
            passed=False,
            expected_output=fn_expected_output,
            actual_output=None)

    # If the types of [output] and [fn_expected_output] don't match, then the test
    # fails. If both are tensors, then check that their dtypes and shapes match as
    # well. Then proceed to check that the values are close enough to pass the test.
    if not isinstance(output, type(fn_expected_output)):
        tqdm.write(f"Function output has type {type(output)}, but expected output has type {type(fn_expected_output)}.")
        return argparse.Namespace(
            passed=False,
            expected_output=fn_expected_output,
            actual_output=output)
    elif isinstance(output, torch.Tensor) and isinstance(fn_expected_output, torch.Tensor):
        if not output.dtype == fn_expected_output.dtype:
            tqdm.write(f"Function output has dtype {output.dtype}, but expected output has dtype {fn_expected_output.dtype}.")
            return argparse.Namespace(
                passed=False,
                expected_output=fn_expected_output,
                actual_output=output)
        elif not output.shape == fn_expected_output.shape:
            tqdm.write(f"Function output has shape {output.shape}, but expected output has shape {fn_expected_output.shape}.")
            return argparse.Namespace(
                passed=False,
                expected_output=fn_expected_output,
                actual_output=output)
        else:
            pass
    
    # This try-except will catch any other issues
    try:
        passed = torch.allclose(output, fn_expected_output)
    except Exception as e:
        passed = False
        tb = traceback.TracebackException.from_exception(e)
        error_trace = "".join(tb.format())
        tqdm.write("Function output could not be compared to expected output. Traceback:")
        tqdm.write(error_trace)
    return argparse.Namespace(
        passed=passed,
        expected_output=fn_expected_output,
        actual_output=output)
    
POSSIBLE_FUNCTIONS = ["get_per_token_logprobs",
    "grpo_rl_objective", "grpo_kl_loss",
    "rewards_to_advantages", "compute_grpo_loss",

    # These are really testing just the output_ids_to_masks() function, but
    # it may be easier to work on its components separately.
    "output_ids_to_attention_mask",
    "output_ids_to_completion_mask",
    "output_ids_to_prompt_mask"]

def load_test_case_data(test_case_data_path):
    """Returns a dictionary mapping function names to lists of test cases.

    Two options: (1) test_case_data.pt: uses torch.load() to get the test case data.
    I only 99.999% trust this to work on arbitrary systems. (2) test_case_data.json:
    custom (de)serialization functions in UtilsBase.py to reconstruct tensor data.
    """
    if test_case_data_path.endswith(".pt"):
        tqdm.write(f"Loading test case data from PT file {test_case_data_path}...")
        fname2test_data = torch.load(test_case_data_path)
    elif test_case_data_path.endswith(".json"):
        tqdm.write(f"Loading test case data from JSON file {test_case_data_path}...")
        with open(test_case_data_path, "r+") as f:
            fname2test_data = json.load(f)
            fname2test_data = UtilsBase.from_json_encoding(fname2test_data)
    else:
        raise ValueError(f"Unknown test_case_data_path extension for {test_case_data_path}")
    return fname2test_data

def format_pass_rate(passed, total):
    """Returns a string like '7/8 (88%)', with the percentage rounded to two
    significant figures.
    """
    percent = 100 * passed / total if total > 0 else 0
    return f"{passed}/{total} ({float(f'{percent:.2g}'):g}%)"

def run_tests(*, fns_to_test, fname2test_data, fname2fn=None, device="cpu",
    verbose=False, verbose2=False):
    """Runs the test cases in [fname2test_data] for each function named in
    [fns_to_test], and returns a Namespace with the number of passed and total tests,
    overall and per function.

    Args:
    fns_to_test     -- names of functions to test
    fname2test_data -- dictionary mapping function names to lists of test cases
    fname2fn        -- dictionary mapping function names to the functions to test, eg.
                        those from another copy of this file. Functions missing from it
                        fail all their tests. If None, tests the functions in this file.
    device          -- device to use for testing
    verbose         -- whether to print expected and received outputs
    verbose2        -- whether to also print the inputs of each test case
    """
    fname2fn = {fn_name: globals()[fn_name] for fn_name in fns_to_test} if fname2fn is None else fname2fn
    verbose = verbose or verbose2

    fname2results = dict()
    failed_tests, total_tests = 0, 0
    for fn_name in fns_to_test:
        test_cases = fname2test_data[fn_name]
        fn = fname2fn.get(fn_name, None)

        tqdm.write(f"Function={fn_name}: Running {len(test_cases)} test cases...")
        if fn is None:
            tqdm.write(f"\tFunction={fn_name} is not defined; all {len(test_cases)} test cases FAILED")
            fname2results[fn_name] = argparse.Namespace(passed=0, total=len(test_cases))
            failed_tests += len(test_cases)
            total_tests += len(test_cases)
            continue

        fn_failed_tests = 0
        for idx,test_case in enumerate(test_cases):
            total_tests += 1
            # Deepcopy data so that nothing can be modified by tested functions
            fn_args = copy.deepcopy(test_case["fn_args"])
            fn_kwargs = copy.deepcopy(test_case["fn_kwargs"])
            fn_expected_output = copy.deepcopy(test_case["fn_expected_output"])

            fn_result = test_fn(fn=fn, fn_args=fn_args, fn_kwargs=fn_kwargs, fn_expected_output=fn_expected_output, device=device, verbose=verbose)
            fn_failed_tests += int(not fn_result.passed)
            test_result_str = "PASSED" if fn_result.passed else "FAILED"
            
            if not verbose and not verbose2:
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
            elif verbose and not verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tExpected output:\n{fn_result.expected_output}")
                tqdm.write(f"\tGot output:\n{fn_result.actual_output}")
            elif verbose2:
                tqdm.write(f"\t=======================================================")
                tqdm.write(f"\tFunction={fn_name} Test case {idx+1}/{len(test_cases)} {test_result_str}")
                tqdm.write(f"\tFunction positional arguments (total={len(fn_args)}):")
                for arg_idx,a in enumerate(fn_args):
                    print_newline = "\n" if isinstance(a, torch.Tensor) and a.ndim > 1 else ""
                    tqdm.write(f"Arg {arg_idx}: {print_newline}{a}")
                tqdm.write(f"\tFunction keyword arguments (total={len(fn_kwargs)}):")
                for k,v in fn_kwargs.items():
                    print_newline = "\n" if isinstance(v, torch.Tensor) and v.ndim > 1 else ""
                    tqdm.write(f"{k}={print_newline}{v}")
                tqdm.write(f"\t=== Expected output:\n{fn_result.expected_output}")
                tqdm.write(f"\t=== Received output:\n{fn_result.actual_output}")

        failed_tests += fn_failed_tests
        fname2results[fn_name] = argparse.Namespace(passed=len(test_cases) - fn_failed_tests, total=len(test_cases))
        
    tqdm.write(f"Testing complete: {format_pass_rate(total_tests-failed_tests, total_tests)} tests passed.")
    return argparse.Namespace(passed=total_tests - failed_tests, total=total_tests,
        fname2results=fname2results)

def get_args(argv=None):
    """Returns parsed command-line arguments for testing."""
    P = argparse.ArgumentParser()
    P.add_argument("--fns_to_test", choices=POSSIBLE_FUNCTIONS, nargs="+",
        default=POSSIBLE_FUNCTIONS,
        help="List of functions to test.")
    P.add_argument("--device", type=str, default="cpu",
        help="Device to use for testing.")
    P.add_argument("--test_case_data_path", type=str,
        default=osp.join(osp.dirname(osp.abspath(__file__)), "test_case_data.json"),
        help="Path to the test case data (.json or .pt)")
    P.add_argument("--seed", type=int, default=42,
        help="Random seed for testing case generation.")
    P.add_argument("-v", "--verbose", action="store_true",
        help="Whether to print verbose output during testing.")
    P.add_argument("-vv", "--verbose2", action="store_true",
        help="Whether to print extra verbose output during testing.")
    args = P.parse_args(argv)
    args.verbose = args.verbose or args.verbose2
    return args

def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    fname2test_data = load_test_case_data(args.test_case_data_path)
    return run_tests(fns_to_test=args.fns_to_test,
        fname2test_data=fname2test_data,
        device=device,
        verbose=args.verbose,
        verbose2=args.verbose2)

if __name__ == "__main__":
    main()
