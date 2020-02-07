
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
ro.r('''.libPaths('C://Users//sugan//Documents//R//win-library//3.5')''')

spatstat=importr("spatstat")

# factor = ro.Vector([1.1, 2.2, 3.3])
res = spatstat.CDF(1.1)
print(res)

# from rpy2.rinterface import RRuntimeError
# from rpy2.robjects.packages import importr
# utils = importr('utils')
#
# def importr_tryhard(packname, contriburl):
#     try:
#         rpack = importr(packname)
#     except RRuntimeError:
#
#         utils.install_packages(packname, contriburl = contriburl, dependencies=True, type="source")
#         rpack = importr(packname)
#     return rpack
# importr_tryhard("spatstat","https://www.rdocumentation.org/packages/spatstat")