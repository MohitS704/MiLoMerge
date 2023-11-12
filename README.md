# bin_merging

This package stores the approximation algorithm within the ```brunelle_merger``` folder.
Tests for how to use the file, as well as processing files for the analytic tests are stored in ```tests``` and ```test_data``` specifically.

## Using the package

At the top of the file that you are using the merger in, type the following at the top:

```python
import sys
sys.path.append("path/to/brunelle_merger/")
import brunelle_merger as bm
import SUPER_ROC_Curves as ROC
```

This will allow you to import the installed package from anywhere on your machine.
