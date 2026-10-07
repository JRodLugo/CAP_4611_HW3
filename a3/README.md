# Assignment 3 report

The report source is `a3.tex`. Compile from this directory with a PDF-capable LaTeX engine. Keep `figs/` alongside the source. Code listings are embedded in the document, so compilation does not require Python or shell escape.

Before submission, replace the name and student-number fields on page one. For a group submission, include both students. Export a single PDF named `a3_LASTNAME.pdf` or `a3_LASTNAME1_LASTNAME2.pdf`, as required by the submission instructions. Review the assistance disclosures at the start of each question.

The Codex preview compiler currently fails before processing the source with `Unable to find standard directories for platform`; final document compilation and page layout have not been verified.

## Reproduce the experiments

Python dependencies: NumPy, SciPy, Matplotlib. From `code/`, run:

```text
python main.py all
```

This generates the figures referenced in the report, plus the starter demonstration figures and the optional polynomial-fit grid. The report includes the requested error curve, measured MSE values, and the relevant implemented code rather than the entire starter framework.

From the repository root, `python verify_assignment.py` checks the weighted and bias normal equations, polynomial basis, robust gradient, and the 100-update optimizer comparison. `results.json` contains selected numerical outputs.

The polynomial implementation uses the requested raw monomial basis and `numpy.linalg.lstsq`. High-degree numerical instability is documented in the report. During the forced 100-update comparison, the supplied line-search optimizer can emit a division warning after convergence; its existing safeguard resets the trial step and all recorded objective values remain finite.
