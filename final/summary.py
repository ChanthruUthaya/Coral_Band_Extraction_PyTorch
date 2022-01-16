class Summary:

  def __init__(self, path):
      self.path = path
      self.df = pd.DataFrame({'epoch':[], 'loss':[], 'max':[], 'average':[]})
      self.start = True
    
  def write_stats(self, epoch, loss, max, average):
    data = [epoch, loss, max, average]
    data_write = "|".join(map(str, data))
    data_df = df.DataFrame({list(self.df)[i]:[data[i]] for i in range(len(data))})
    if self.start:
      mode = 'w'
      self.start = False
    else:
      mode = 'a'
    
    with open(self.path, mode) as f:
      f.write(data_write)
      f.write("\n")
    
    self.df.append(data_df, ignore_index=True)

  def visualise_stats(self, )


    
#TODO: 
  #VISUALISE BOTH METRICS