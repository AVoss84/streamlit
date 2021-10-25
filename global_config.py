import os, sys

#-------------------------------
# Which dev environment to use?
#-------------------------------

using = 'vm'                    # own virtual machine
#using = 'docker'                # Docker container

## Check if required environment variables exist
## if not apply default paths from test environment:
#-----------------------------------------------------------
if using == 'vm':
    defaults = {"UC_CODE_DIR": "/home/G126143/claims-dashboard/",                
                #"UC_LOG_DIR": "/home/G126143/fraud-model-dev/src/log-files/",        
                "UC_DATA_ROOT": "/data/data/claims_reporting/input_data/",       
                #"UC_DATA_ROOT": "/data/data/fraud_model/",
                #"UC_SOURCE_DATA_ROOT" : "/data/data/fraud_model/",       # only needed in case of MFT creation! Hence not for Docker.
                "UC_PROFILE" :"prod",        # "prod" ; set this profile for overall Postgres usage, also Flask server
                "UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz:5432'
                #"UC_PORT":"5000", 
                #"UC_APP_CONNECTION": '0.0.0.0'
            }
else:
    defaults = {
            "UC_CODE_DIR": "/app/src/",                
            #"UC_LOG_DIR": "/app/src/log-files/",           
            "UC_DATA_ROOT": "/app/Data/claims_reporting/input_data",         
            #"UC_SOURCE_DATA_ROOT" : "/app/Data/fraud_model/",     # just for completeness!       
            "UC_PROFILE" :"prod",        # "prod" ; set this profile for overall Postgres usage, also Flask server
            "UC_DB_CONNECTION": 'postgresql://postgres:kakYritiven@agcs-postgres-1-server.service.dsp.allianz:5432'
            #"UC_PORT":"5000", 
            #"UC_APP_CONNECTION": '0.0.0.0'
            }                      

#-------------------------------------------------------------------------------------------------------------------------------

env_list = ["UC_DATA_ROOT",
            'UC_CODE_DIR',  "UC_PROFILE", 
            'UC_DB_CONNECTION'
            ]

for env in env_list:
    #if env not in os.environ:
        os.environ[env] = defaults[env]
        print("Environment Variable: " + str(env) + " has been set to default: " + str(os.environ[env]))
          
UC_CODE_DIR = os.environ['UC_CODE_DIR']  
#UC_LOG_DIR = os.environ["UC_LOG_DIR"]
UC_PROFILE = os.environ['UC_PROFILE']
#UC_PORT = os.environ['UC_PORT']
UC_DB_CONNECTION = os.environ['UC_DB_CONNECTION'] +'/'+ UC_PROFILE    # for the global default usage in the package
UC_DATA_ROOT = os.environ['UC_DATA_ROOT']
