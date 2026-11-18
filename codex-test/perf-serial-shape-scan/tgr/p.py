import os
import sys
from pathlib import Path
import driver

case = Path(os.environ["TGR_CASE"])
out = Path(os.environ["TGR_ROUTE_OUTPUT"])
checked = Path(os.environ["TGR_PRECISION"])
route = os.environ["TGR_ROUTE"]
sys.argv = ["driver.py", "profile", str(case), str(out), "--route", route, "--physical", "10", "--precision", str(checked)]
raise SystemExit(driver.main())
