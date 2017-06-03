-- adapated from https://john.nachtimwald.com/2014/08/06/using-lua-as-a-templating-engine/
--
--- Template renderer.
--
-- Takes a string with embedded Lua code and renders
-- it based on the Lua code.
--
-- Supports:
--  * {{var}} for printing variables.
--  * {% func %} for running Lua functions.
--
--  Multi-line strings in Lua blocks are supported but
--  [[ is not allowed. Use [=[ or some other variation.
--
--  Both compile and compile_file can take an optional
--  env table which when provided will be used as the
--  env for the Lua code in the template. This allows
--  a level of sandboxing. Note that any globals including
--  libraries that the template needs to access must be
--  provided by env if used.
 
local M = {}
 
-- Append text or code to the builder.
local function appender(builder, text, code)
    if code then
        builder[#builder+1] = code
    else
        -- [[ has a \n immediately after it. Lua will strip
        -- the first \n so we add one knowing it will be
        -- removed to ensure that if text starts with a \n
        -- it won't be lost.
        builder[#builder+1] = "_ret[#_ret+1] = [[\n" .. text .. "]]"
    end
end
 
--- Takes a string and determines what kind of block it
-- is and takes the appropriate action.
--
-- The text should be something like:
-- "{{ ... }}"
-- 
-- If the block is supported the begin and end tags will
-- be stripped and the associated action will be taken.
-- If the tag isn't supported the block will be output
-- as is.
local function run_block(builder, text)
    local func
    local tag
 
    tag = text:sub(1, 2)
 
    if tag == "{{" then
        func = function(code)
            return ('_ret[#_ret+1] = %s'):format(code)
        end
    elseif tag == "{%" then
        func = function(code)
            return code
        end
    end
    if func then
        appender(builder, nil, func(text:sub(3, #text-3)))
    else
        appender(builder, text)
    end
end
 
--- Compile a Lua template into a string.
--
-- @param      tmpl The template.
-- @param[opt] env  Environment table to use for sandboxing.
--
-- return Compiled template.
function M.compile(tmpl)
    -- Turn the template into a string that can be run though
    -- Lua. Builder will be used to efficiently build the string
    -- we'll run. The string will use it's own builder (_ret). Each
    -- part that comprises _ret will be the various pieces of the
    -- template. Strings, variables that should be printed and
    -- functions that should be run.
    local builder = { "_ret = {}\n" }
    local pos     = 1
    local b
    local func
    local err

    if #tmpl == 0 then
        return ""
    end
 
    while pos < #tmpl do
        -- Look for start of a Lua block.
        doublebracepos = tmpl:find("{{", pos)
        bracepercentpos = tmpl:find("{%%", pos)
        if doublebracepos and not bracepercentpos or ( doublebracepos and bracepercentpos and doublebracepos < bracepercentpos ) then 
            b = doublebracepos
            -- Add all text up until this block.
            appender(builder, tmpl:sub(pos, b-1))
            -- Find the end of the block.
            pos = tmpl:find("}}", b)
            if not pos then
                appender(builder, "End tag ('}}') missing")
                break
            end
            run_block(builder, tmpl:sub(b, pos+2))
            -- Skip back the }} (pos points to the start of }}).
            pos = pos+2
        elseif bracepercentpos then
            b = bracepercentpos
            appender(builder, tmpl:sub(pos, b-1))
            -- Find the end of the block.
            pos = tmpl:find("%%}", b)
            if not pos then
                appender(builder, "End tag ('%}') missing")
                break
            end
            run_block(builder, tmpl:sub(b, pos+2))
            -- Skip back the }} (pos points to the start of }}).
            pos = pos+2
        else
            break
        end
    end
    -- Add any text after the last block. Or all of it if there
    -- are no blocks.
    if pos then
        appender(builder, tmpl:sub(pos, #tmpl-1))
    end
 
    builder[#builder+1] = "return table.concat(_ret)"
    local tc = table.concat(builder, "\n")
    local func, err = loadstring(tc)
    if not func then
        return err
    end
    return func()
end
 
function M.compile_file(name)
    local f, err = io.open(name, "rb")
    if not f then
        return err
    end
    local t = f:read("*all")
    f:close()
    return M.compile(t)
end
 
return M

