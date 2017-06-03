color = 'yellow'
f, err = loadstring('print "hi"; print("color", color ); return 3')
if not f then
    print('err', err)
end
color = 'pink'
v = f()
print('v', v)

