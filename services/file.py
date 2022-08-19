import pandas as pd
#import sqlalchemy as sql
from imp import reload
import os, yaml, sys, stat, pickle, json
from copy import deepcopy
import dateutil
from pathlib import Path


class CSVService:
    def __init__(self, path="", delimiter="\t", encoding="UTF-8", schema_map=None, root_path: str = '', **kwargs):
        # self.__dict__.update(kwargs)

        self.path = os.path.join(root_path, path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.schema_map = schema_map
        self.kwargs = kwargs

    def doRead(self, **kwargs):
        df = pd.read_csv(filepath_or_buffer=self.path, encoding=self.encoding, 
                         low_memory=False,     # set to avoid dtype warning 
                         delimiter=self.delimiter, **kwargs)
        print("CSV Service read from file: " + str(self.path))
        if self.schema_map != None:
            df.rename(columns=self.schema_map, inplace=True)
        return df

    def doWrite(self, X):
        X.to_csv(path_or_buf=self.path, encoding=self.encoding, sep=self.delimiter, **self.kwargs)
        print("CSV Service Output to File: " + str(self.path))
        try:
            # Set rwx for owner, group, others:
            os.chmod(self.path, stat.S_IRWXG | stat.S_IRWXU | stat.S_IRWXO)
        except Exception as e:
            print(e)     


class XLSXService:
    def __init__(self, path="", sheetname="Sheet1", root_path: str = '', schema_map=None, **kwargs):
        self.path = os.path.join(root_path, path)
        self.writer = pd.ExcelWriter(self.path)
        self.sheetname = sheetname
        self.schema_map = schema_map
        self.kwargs = kwargs

    def doRead(self, **kwargs):
        df = pd.read_excel(self.path, engine='openpyxl', **kwargs)     # added engine option
        print("XLS Service Read from File: " + str(self.path))
        if self.schema_map != None:
            df.rename(columns=self.schema_map, inplace=True)
        return df    
        
    def doWrite(self, X, sheetname="Sheet1"):
        X.to_excel(self.writer, self.sheetname, **self.kwargs)
        self.writer.save()
        print("XLSX Service Output to File: " + str(self.path))
        try:
            # Set rwx for owner, group, others:
            os.chmod(self.path, stat.S_IRWXG | stat.S_IRWXU | stat.S_IRWXO)
        except Exception as e:
            print(e)


class PickleService:
    def __init__(self, path = "", root_path: str = '', id_col = None, schema_map = None, df=True, **kwargs):
        if root_path:
            self.path = os.path.join(root_path, path)
        else:
            self.path = path
        self.id_col = id_col
        self.schema_map = schema_map
        self.kwargs = kwargs
        self.df = df         # boolean: Is object as dataframe?

    def doRead(self, **kwargs):
        if self.df:
            data = pd.read_pickle(self.path, **kwargs)
            print("Pickle Service Read from File: " + str(self.path))
            if self.schema_map != None:
                data.rename(columns = self.schema_map, inplace = True)
        else:
            data = pickle.load(open(self.path, "rb"))
            print("Pickle Service Read from File: " + str(self.path))
        return data

    def doWrite(self, X):
        if self.df:
            X.to_pickle(path = self.path, compression = None)        # "gzip"
        else:
            pickle.dump(X, open(self.path, "wb"))
        print("Pickle Service Output to File: "+ str(self.path))

        # try:
        #     # Set rwx for owner, group, others:
        #     os.chmod(self.path, stat.S_IRWXG | stat.S_IRWXU | stat.S_IRWXO)
        # except Exception as e:
        #     print(e)


# class PostgresService:
#     def __init__(self,
#                  qry: str = None
#                  , method = 'multi'
#                  , if_exists = 'append'
#                  , connection_string = glob.UC_DB_CONNECTION   # ToDo: save in .pgpass file!
#                  , output_tbl : str = 'my_table'
#                  , verbose = True
#                 ):
#         self.qry = qry
#         self.output_tbl = output_tbl
#         self.if_exists = if_exists
#         self.method = method
#         self.connection_string = connection_string
#         self.verbose = verbose
#         self.conn = self._create_connection(self.connection_string)
    
#     def __del__(self):
#         self.conn.dispose()
#         #if self.verbose : print("connection disposed in destructor")
     
#     def _create_connection(self, connection_string):
#         try:
#             engine = sql.create_engine(self.connection_string)
#             if self.verbose : print(f"Connection engine created, using {self.connection_string}")
#         except Exception as e:
#             engine = None
#             print(e)
#             print("Connection could not be established")
#         return engine

#     def getTypes(self, table_name):
#         metadata = sql.MetaData()
#         table = sql.Table(table_name, metadata, autoload=True, autoload_with=self.conn)
#         columns = table.c
#         dtypes = {}
#         for c in columns:
#             dtypes[c.name] = c.type
#         return dtypes    


#     def doRead(self, **other):
#         """Send (any) query - except of insert - to PG server"""
#         if not self.qry:
#             print("Error - no SQL query string provided")
#             return False
#         if self.conn is None:
#             print("Error - No Connection available")
#             return False
#         else:
#             try:
#                self.selected_tables = pd.read_sql_query(self.qry, self.conn, **other)
#                if self.verbose : print(f"Query successful.")
#                return self.selected_tables
#             except IOError as e:
#                print(e)
#                print(f"Query not successful!!")
#                return None

#     def doWrite(self, X : pd.DataFrame, **others):
#         """Write dataframe to Postgres"""
#         dat = deepcopy(X)
#         try:
#             dat.to_sql(self.output_tbl, con=self.conn, if_exists = self.if_exists, method = self.method, index=False, **others)
#             if self.verbose : print(f"Data successfully written to {self.output_tbl}")
#         except IOError as e:
#             print(e)
#             print("Could not write to database!")   

                    
class YAMLservice:
        def __init__(self, child_path : str = '', 
                     root_path = Path.cwd(), verbose = False, **kwargs):
            
            self.root_path = root_path
            self.child_path = child_path
            self.verbose = verbose
            self.kwargs = kwargs
        
        def doRead(self, filename = None, **kwargs):  
            """
            Read in YAMl file from specified path
            """
            with open(self.root_path / self.child_path / filename , 'r') as stream:
                try:
                    my_yaml_load = yaml.safe_load(stream)
                    if self.verbose: print(f"Read: {self.root_path / self.child_path / filename}")
                    return my_yaml_load    
                except yaml.YAMLError as exc:
                    print(exc) 
            

        def doWrite(self, X, filename = None):
            """
            Write dictionary X to YAMl file
            """
            with open(self.root_path / self.child_path / filename, 'w') as outfile:
                try:
                    yaml.dump(X, outfile, default_flow_style = False, **self.kwargs)
                    if self.verbose: print(f"Write to: {self.root_path / self.child_path / filename}")
                    # try:
                    #     # Set rwx for owner, group, others:
                    #     os.chmod(self.path, stat.S_IRWXG | stat.S_IRWXU | stat.S_IRWXO)
                    # except Exception as e:
                    #     print(e)    
                except yaml.YAMLError as exc:
                    print(exc) 


# class TXTService:
#     def __init__(self, path="myfile", encoding="utf-8", root_path=glob.UC_DATA_DIR, **kwargs):
#         self.myfile = path        # filename
#         self.myfolder = root_path          # folder/subfolder
#         self.path = os.path.join(root_path, path)
#         self.encoding = encoding
#         self.kwargs = kwargs
#         self.file_client = service_client.get_file_client(file_system=glob.UC_FILESYSTEM, file_path=self.path) 

#     def doRead(self, **kwargs) -> list:
#         try:
#             download = self.file_client.download_file()
#             download_bytes = download.readall()
#             assert self.file_client.get_file_properties().size > 0, 'File has size zero.'
#             df = download_bytes.decode(self.encoding).splitlines() 
#             print("TXT Service read from file: " + str(self.path))    
#         except Exception as e0:
#             print(e0); df = None
#         finally: 
#             return df
        
#     def doWrite(self, X : list):
#         try:
#             data = '\n'.join(X).encode(self.encoding)
#             self.file_client.create_file()
#             self.file_client.append_data(data, offset=0, length=len(data))
#             self.file_client.flush_data(len(data))
#             print("TXT Service output to file: " + str(self.path))  
#             return True
#         except Exception as e0:
#             print(e0); return False


class JSONservice:
        def __init__(self, child_path : str = '', 
                     root_path = Path.cwd(), verbose = True, **kwargs):
            
            self.root_path = root_path
            self.child_path = child_path
            self.verbose = verbose
            self.kwargs = kwargs
        
        def doRead(self, filename : str, **kwargs):  
            """
            Read in JSON file from specified path
            """
            try:
                with open(self.root_path / self.child_path / filename , 'r') as stream:
                    my_json_load = json.load(stream)                    
                if self.verbose: print(f"Read: {self.root_path / self.child_path / filename}")
                return my_json_load    
            except Exception as exc:
                print(exc) 
            
        def doWrite(self, X: dict, filename : str):
            """
            Write X to JSON file
            """
            with open(self.root_path / self.child_path / filename, 'w', encoding='utf-8') as outfile:
                try:
                    json.dump(X, filename, ensure_ascii=False, indent=4, **self.kwargs)
                    if self.verbose: print(f"Write to: {self.root_path / self.child_path / filename}")
                except Exception as exc:
                    print(exc) 