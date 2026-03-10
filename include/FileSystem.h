//
//  FileSystem.h
//  main
//
//

#ifndef FileSystem_h
#define FileSystem_h

#include <filesystem>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>


#include "MPIInfo.h"

/**
 * This class creates folders for output by a relative path.
 *
 */
class FileSystem
{
public:
    struct SubDir
    {
        const std::string subDir;
        std::string&      prmDir;
        
        SubDir(const std::string& subDir,
               std::string&      prmDir);
        
        };
        
    
    // return present working directory
    static std::string pwd();
    
    
    
    // check and create directories:
    //  - must be a relative directory
    //  - target directory is the under pwd()
    //  - sucess: return true
    //  - invalid / fail: return false
    static bool dir(const std::string& rel_dir);
    
    
    
    // check and create directories:
    //  - must be a relative directory
    //  - target directory is the under pwd()
    //  - sucess: return true
    //  - invalid / fail: return false
    static bool outputDirSystem(const MPIInfo& mpiInfo,
                                std::string& rel_dir,
                                std::string& sub_dir_path,
                                std::vector<SubDir>& sub_dirs);


    
    static std::string find_next_numeric_subdir(const std::string &path_str);
    
    static bool numeric_subdir(std::string& subdir,
                               const std::string& rel_dir);
    
private:
    // varify if child is under or equal to base
    static bool __is_subpath(const std::filesystem::path& base,
                             const std::filesystem::path& child);
    
    
    // varify if the string is all digits
    static bool __is_all_digits(const std::string& name);
    
    
    
};




#endif /* FileSystem_h */
