import os, sys
from pathlib import Path

#-------------------------------
# Which dev environment to use?
#-------------------------------

using = 'vm'                    # own virtual machine
#using = 'docker'                # Docker container

## Check if required environment variables exist
## if not apply default paths from test environment:
#-----------------------------------------------------------
if using == 'vm':
    defaults = {"UC_CODE_DIR": str(Path.home() / "Documents/GitHub/streamlit/src/"),                      
                "UC_DATA_ROOT": str(Path.home() / "Documents/GitHub/streamlit/data/"),       
                "UC_PROFILE" :"prod",        # "prod" ; set this profile for overall Postgres usage, also Flask server
                #"UC_DB_CONNECTION": '',
                #"UC_PORT":"5000", 
                #"UC_APP_CONNECTION": '0.0.0.0'
                "UC_OPENAI_API_KEY": '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'
            }
else:
    defaults = {
            "UC_CODE_DIR": "/app/src/",                          
            "UC_DATA_ROOT": "/app/Data/reporting/data",            
            "UC_PROFILE" :"prod",        # "prod" ; set this profile for overall Postgres usage, also Flask server
            #"UC_DB_CONNECTION": '',
            #"UC_PORT":"5000", 
            #"UC_APP_CONNECTION": '0.0.0.0'
            "UC_OPENAI_API_KEY": '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'
            }                      

#-------------------------------------------------------------------------------------------------------------------------------

env_list = ["UC_DATA_ROOT",
            'UC_CODE_DIR',  "UC_PROFILE", "UC_OPENAI_API_KEY"
            #'UC_DB_CONNECTION'
            ]

for env in env_list:
    #if env not in os.environ:
        os.environ[env] = defaults[env]
        print("Environment Variable: " + str(env) + " has been set to default: " + str(os.environ[env]))
          
UC_CODE_DIR = os.environ['UC_CODE_DIR']  
#UC_LOG_DIR = os.environ["UC_LOG_DIR"]
UC_PROFILE = os.environ['UC_PROFILE']
#UC_PORT = os.environ['UC_PORT']
#UC_DB_CONNECTION = os.environ['UC_DB_CONNECTION'] +'/'+ UC_PROFILE    # for the global default usage in the package
UC_DATA_ROOT = os.environ['UC_DATA_ROOT']
UC_OPENAI_API_KEY = os.environ['UC_OPENAI_API_KEY']