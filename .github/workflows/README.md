# CI/CD Workflow Documentation

## Overview

The GitHub Actions workflow (`.github/workflows/qa.yml`) provides automated quality assurance for the blink detection evaluation system.

## What it does

### Triggers
- **Pull Requests** to `main` or `master` branch
- **Direct pushes** to `main` or `master` branch

### Steps

1. **Environment Setup**
   - Checkout code
   - Set up Python 3.12
   - Cache pip dependencies for faster builds
   - Install dependencies from `requirements.txt`

2. **Evaluation Suite**
   - Runs the complete evaluation harness
   - Processes all videos (original + augmented)
   - Computes metrics and threshold validation
   - Generates structured results

3. **Results Processing**
   - Creates human-readable summary in Markdown
   - Uploads results as build artifacts
   - Comments on PRs with evaluation results

4. **Quality Gates**
   - Validates metrics against defined thresholds
   - Fails the build if thresholds are not met
   - Provides detailed feedback on violations

## Artifacts

The workflow generates several artifacts:

### `evaluation-results`
- Complete evaluation results directory
- All JSON files and detailed metrics
- Human-readable summary

### `metrics-json`
- Standalone `metrics.json` file
- Structured evaluation results
- Can be used for further analysis

## PR Comments

When a PR is created or updated, the workflow automatically:

1. Runs the evaluation suite
2. Generates a summary of results
3. Posts a comment with:
   - Key metrics (precision, recall, F1-score, etc.)
   - Threshold validation status
   - Links to build artifacts

## Threshold Validation

The workflow includes automatic threshold checking:

- **Pass**: All metrics meet minimum requirements
- **Fail**: One or more metrics fall below thresholds
- **Warning**: Metrics are below target but above minimum

### Failure Conditions
- Precision < 0.4
- Recall < 0.8
- F1-Score < 0.5
- Accuracy < 0.6
- Temporal accuracy < 0.9
- Videos evaluated < 8

## Local Testing

To test the workflow locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
cd evaluations
python evaluation_harness.py

# Check thresholds
python threshold_checker.py --metrics results/metrics.json
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**
   - Ensure `requirements.txt` is up to date
   - Check that all packages are available

2. **Video files not found**
   - Ensure video files are in the correct locations
   - Check file paths in the evaluation harness

3. **Threshold failures**
   - Review current metrics vs. thresholds
   - Adjust thresholds in `evaluations/thresholds.json` if needed

4. **Memory issues**
   - Large video files may cause memory problems
   - Consider using smaller test videos for CI/CD

### Debugging

1. **Check workflow logs**
   - Go to Actions tab in GitHub
   - Click on the failed workflow
   - Review step-by-step logs

2. **Download artifacts**
   - Even failed builds generate artifacts
   - Download and inspect results locally

3. **Run locally**
   - Clone the repository
   - Follow local testing steps above

## Configuration

### Thresholds
Edit `evaluations/thresholds.json` to adjust quality gates:

```json
{
  "thresholds": {
    "blink_detection": {
      "average_precision": {"minimum": 0.4},
      "average_recall": {"minimum": 0.8}
    }
  }
}
```

### Workflow Triggers
Modify the `on` section in `.github/workflows/qa.yml`:

```yaml
on:
  pull_request:
    branches: [ main, master ]
  push:
    branches: [ main, master ]
  # Add more triggers as needed
```

## Best Practices

1. **Keep thresholds realistic**
   - Set based on actual performance
   - Review and adjust regularly

2. **Monitor build times**
   - Large videos increase build time
   - Consider using smaller test videos

3. **Review results regularly**
   - Check PR comments for trends
   - Monitor threshold violations

4. **Update dependencies**
   - Keep requirements.txt current
   - Test locally before pushing

## Support

For issues with the CI/CD workflow:

1. Check the workflow logs
2. Review the troubleshooting section
3. Test locally to isolate issues
4. Update documentation as needed 