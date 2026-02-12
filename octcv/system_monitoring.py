import re
import pandas as pd
import psutil
import asyncio
import time
from IPython.display import display, update_display, HTML


def isNumericString(numstrcand: str):
    """
    Validates if a string is a formatted or unformatted numeric representation.
    
    The function uses a regular expression to match integers (with or without 
    thousands-separator commas) and optional decimal fractions (including 
    leading decimals like '.25').

    ### Regex Breakdown:
    `^([0-9]{1,3}(,?[0-9]{3})*)?([.][0-9]+)?$`

    * `^` and `$` : Anchors the match to the start and end of the string.
    * `([0-9]{1,3}(,?[0-9]{3})*)?` : **Optional Integer Group**
        * `[0-9]{1,3}` : Matches the leading 1 to 3 digits (e.g., '1', '12', or '123').
        * `(,?[0-9]{3})*` : Matches zero or more groups of three digits. 
            * The `,?` makes the comma optional, allowing both '1,000' and '1000'.
    * `([.][0-9]+)?` : **Optional Decimal Group**
        * Matches a literal dot followed by one or more digits. Because the 
            Integer Group is optional, this allows strings starting with '.' (e.g., '.5').

    ### Valid Examples:
    * Integers: "123", "1234", "1,234"
    * Decimals: "123.45", "1,234.56", ".25"
    * Large Numbers: "1000000", "1,000,000"

    Args:
        numstrcand (str): The string candidate to be checked.

    Returns:
        bool: True if the string matches the numeric pattern, False otherwise.
    """
    pattern = r"^([0-9]{1,3}(,?[0-9]{3})*)?([.][0-9]+)?$"
    if re.match(pattern, numstrcand):
        return True
    else:
        return False

def numericString2Numeric(numericString):
    if isNumericString(numericString):
        numericString = numericString.replace(',','')
        try:
            return int(numericString)
        except:
            try:
                return float(numericString)
            except Exception as e:
                raise(e)
    else:
        raise ValueError(f"{numericString} is not properly formatted.")

def SizeStr2IntBytes(size):
    if isinstance(size,(int,float)):
        return float(size)
    
    pattern = r'([[0-9].,]+)\s*([a-zA-Z]*)'
    rematch = re.match(pattern, size)
    if not rematch:
        raise ValueError(f"Improper format for provided size argument: {size}")
    
    num,units = rematch.groups()

    num = numericString2Numeric(num)
    
    units = 'bytes' if not units else units

    unitMap = {
        1 * 8**-1 : ['','','bits'],
        1 : ['B','','bytes'],
        1e3 : ['kB','kiB','kilobytes'],
        1e6 : ['MB','MiB','megabytes'],
        1e9 : ['GB','GiB','gigabytes'],
        1e12 : ['TB','TiB','terabytes'],
        1e15 : ['PB','PiB','petabytes'],
        1e18 : ['EB','EiB','exabytes'],
        1e21 : ['ZB','ZiB','zettabytes']
    }

    conversion_factor = None
    for cf,accepted_units in unitMap.items():
        for au in accepted_units:
            if au.lower() == units.lower():
                conversion_factor = cf
                break
        if conversion_factor is not None:
            break

    n_bytes = num * conversion_factor
    
    return n_bytes        
        
def hrByteSizeStr(n_bytes,precision=2,unit_format=0):
    if n_bytes > 1e21:
        raise ValueError(f"size > 1 zettabyte not accepted")
        
    unitMap = {
        1 * 8**-1 : ['b','','bits'],
        1 : ['B','','bytes'],
        1e3 : ['kB','kiB','kilobytes'],
        1e6 : ['MB','MiB','megabytes'],
        1e9 : ['GB','GiB','gigabytes'],
        1e12 : ['TB','TiB','terabytes'],
        1e15 : ['PB','PiB','petabytes'],
        1e18 : ['EB','EiB','exabytes'],
        1e21 : ['ZB','ZiB','zettabytes']
    }
    
    n_bytes = SizeStr2IntBytes(n_bytes)
    integer_part = str(n_bytes).split('.')[0]
    integer_ndigits = len(integer_part)
    conversion_factors = list(unitMap.keys())
    accepted_units = list(unitMap.values())

        
    for i in range(len(unitMap)-1):
        if n_bytes >= conversion_factors[i] and n_bytes < conversion_factors[i+1]:
            hrnum = n_bytes / conversion_factors[i]
            hrunit = accepted_units[i][unit_format]
            break
    
    return f"{hrnum:.{precision}f} {hrunit}"
    
def memory_report(output_format=None,human_readable=True):
    vmem = psutil.virtual_memory()
    if output_format is None:
        return vmem

    # Establish field-value map for memory attributes, 
    # starting with attributes common to most operating systems
    vmemMap = dict(
        free = vmem.free,
        used = vmem.used,
        total = vmem.total,
        available = vmem.available,
        percent = vmem.percent,
    )

    # Attempt UNIX-specific attributes
    try:
        vmemMap['active'] = vmem.active
        vmemMap['inactive'] = vmem.inactive
    except:
        pass

    # Convert to human-readable format if desired
    if human_readable:
        memdata = { k:[hrByteSizeStr(v)] for k,v in vmemMap.items() if k != 'percent'}
        memdata['percent'] = [str(vmem.percent) + ' %']
        

    # Parse output format
    match output_format:
        case 'df' | 'dataframe' :
            return pd.DataFrame(memdata)
        case 'md' | 'markdown' :
            return pd.DataFrame(memdata).to_markdown(index=False)
        case 'dict' | 'dictionary' :
            return vmemMap
        case 'p' | 'print' | 'printout':
            for k,v in vmemMap.items():
                print(k.title(),':',f"{v:,f}")
        case _:
            raise ValueError(f"Invalid output_format: {output_format}")
            
# async def background_monitor():
#     # Create a display handle with an initial ID
#     display_id = "mem_monitor"
#     header = HTML("<b>Live Memory Usage:</b> <span id='mem_val'>Starting...</span>")
#     display(header, display_id=display_id)

#     while True:
#         memdf = memory_report(output_format='df',human_readable=True)
#         u,a,t,p = memdf[['used','available','total','percent']].values[0]
        
#         # Update the existing display handle
#         update_display(
#             HTML(f"<b>Live Memory Usage:</b> <code>{u} / {t} ({percent}%)</code><br>{a} Available"), 
#             display_id=display_id
#         )
#         await asyncio.sleep(1)
        
# async def background_memory_monitor():
#     # Create a display handle with an initial ID
#     display_id = "mem_monitor"
#     header = HTML("<b>Live Memory Usage:</b> <span id='mem_val'>Starting...</span>")
#     display(header, display_id=display_id)

#     while True:
#         mem_percent = psutil.virtual_memory().percent
#         a = psutil.virtual_memory().available
#         a = hrByteSizeStr(a)
#         u = psutil.virtual_memory().used
#         u = hrByteSizeStr(u)
#         t = psutil.virtual_memory().total
#         t = hrByteSizeStr(t)
        
#         # Update the existing display handle
#         html = f"""
#         <b>Live Memory Usage:</b> \t <code> {u} / {t} ({mem_percent}%)</code>
#         """
#         update_display(
#             HTML(html), 
#             display_id=display_id
#         )
#         await asyncio.sleep(1)

# async def background_memory_monitor():
#     # Create a display handle with an initial ID
#     display_id = "mem_monitor"
#     header = HTML("<b>Live Memory Usage:</b> <span id='mem_val'>Starting...</span>")
#     display(header, display_id=display_id)

#     while True:
#         mem_percent = psutil.virtual_memory().percent
#         a = psutil.virtual_memory().available
#         a = hrByteSizeStr(a)
#         u = psutil.virtual_memory().used
#         u = hrByteSizeStr(u)
#         t = psutil.virtual_memory().total
#         t = hrByteSizeStr(t)
        
#         # Update the existing display handle
#         update_display(
#             HTML(f"<b>Live Memory Usage:</b> \t <code> {u} / {t} ({mem_percent}%)</code>"), 
#             display_id=display_id
#         )
#         await asyncio.sleep(1)

async def background_memory_monitor():
    display_id = "mem_monitor"
    # Initial placeholder
    display(HTML("<b>Initializing Monitor...</b>"), display_id=display_id)

    while True:
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        u = hrByteSizeStr(mem.used)
        t = hrByteSizeStr(mem.total)
        
        # Determine bar color based on usage intensity
        if mem_percent < 70:
            color = "#28a745" # Green
        elif mem_percent < 90:
            color = "#ffc107" # Yellow
        else:
            color = "#dc3545" # Red

        # Inline HTML/CSS for the bar
        bar_html = f"""
        <div style="display: flex; align-items: center; font-family: monospace;">
            <b style="margin-right: 10px;">RAM:</b>
            <div style="background-color: #e9ecef; border-radius: 4px; width: 200px; height: 18px; margin-right: 10px; overflow: hidden; border: 1px solid #ccc;">
                <div style="background-color: {color}; width: {mem_percent}%; height: 100%; transition: width 0.4s ease;"></div>
            </div>
            <code>{u} / {t} ({mem_percent}%)</code>
        </div>
        """
        
        update_display(HTML(bar_html), display_id=display_id)
        await asyncio.sleep(1)

def startMemoryMonitor():
    loop = asyncio.get_event_loop()
    task = background_memory_monitor
    monitor_task = loop.create_task(task())
