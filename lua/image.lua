require 'image'
require 'torch'

local size=256
a = torch.FloatTensor(3,size,size):uniform()
for c=1,3 do
  for i=1,size do
    for j=1,size do
      if c == 1 then
        a[c][i][j] = i/size * j/size
      elseif c == 2 then
        a[c][i][j] = 1 - i/size * j/size
      else
        a[c][i][j] = (1 - i/size) * j/size
      end
    end
  end
end
image.save('~foo.png', a)


