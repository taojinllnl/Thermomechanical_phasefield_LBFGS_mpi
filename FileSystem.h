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

class FileSystem
{
public:

    // return present working directory
    static std::string pwd();
    

    
    // check and creat dir:
    //  - must be a relative directory
    //  - target directory is the under pwd()
    //  - sucess: return true
    //  - invalid / fail: return false
    static bool dir(const std::string& rel_dir);

private:
    // varify if child is under or equal to base
    static bool __is_subpath(const std::filesystem::path& base,
                             const std::filesystem::path& child);
};




#endif /* FileSystem_h */
