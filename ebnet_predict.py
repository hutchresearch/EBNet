import ebnet
from astropy.table import Table

path = "./sample.fits"
data = Table.read(path)
result = ebnet.predict(data)
print(result)