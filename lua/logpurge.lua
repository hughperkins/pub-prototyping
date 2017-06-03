require 'os'
require 'sys'
require 'paths'

sys.execute('mkdir /tmp/testlogs')
for i=45,62 do
  sys.execute('touch /tmp/testlogs/model_e' .. i .. '.net')
end

function purge_old_models(savedir, epoch, keep_epoch_mod, filename_filter)
  local model_filenames = sys.execute('cd ' .. savedir .. '; ls -t ' .. filename_filter .. ' 2>/dev/null')
  for model_filename in model_filenames:gmatch('[^\n]+') do
--    print('model_filename [' .. model_filename .. ']')
    local filename_pattern = filename_filter:gsub('*', '(%%d+)')
--    print('filename_pattern', filename_pattern)
    local this_epoch = tonumber(model_filename:gmatch(filename_pattern)())
--    print('epoch', epoch)
    if this_epoch % keep_epoch_mod ~= 0 then
      if this_epoch < epoch then
        print('purge', model_filename)
        sys.execute('rm ' .. savedir .. '/' .. model_filename)
      end
    end
  end
end

purge_old_models('/tmp/testlogs', 58, 50, 'model_e*.net')
os.execute('ls /tmp/testlogs')

