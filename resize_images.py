from PIL import Image
import pandas as pd

data = pd.read_csv('monet_data_w_cluster.csv')
size = 400, 260

for i in range(data.shape[0]):
  if (data.cluster_count[i] != '-1'):
    try:
      outfile = ("images/scale/" + str(data.file_id_full[i])  + ".jpg")
      im = Image.open("images/" + str(data.file_id_full[i])  + ".jpg")

      im.thumbnail(size, Image.ANTIALIAS)
      im.save(outfile, "JPEG")
    except:
      print "cannot create thumbnail for image %d" % (data.file_id_full[i])