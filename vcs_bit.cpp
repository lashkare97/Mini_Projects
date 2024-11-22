#include <iostream>
#include <filesystem>
#include <string>
#include "bit.h"

namespace fs = std::filesystem;

void printHelp() {
    std::cout << "Commands:\n"
        << "1. Commit \"message\" - Commit changes with message\n"
        << "2. List Versions - List all Versions\n"
        << "3. List Changes - List all uncommitted changes\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " /path/to/project\n";
        return 1;
    }

    std::string workingDirectory = argv[1];

    try {
        if (!fs::exists(workingDirectory)) {
            std::cerr << "Error: The specified directory do not exist.\n";
            return 1;
        }

        fs::current_path(workingDirectory);

        if (!fs::exists(".bit")) {
            fs::create_directory(".bit");
        }

        Bit bitSystem(workingDirectory);

        printHelp();

        std::string command;
        while (true) {
            std::cout << "bit> ";
            std::getline(std::cin, command);

            if (command == "exit") {
                break;
            }
            else if (command.rfind("Commit ", 0) == 0) {
                bitSystem.Commit(command.substr(7));
            }
            else if (command == "List Versions") {
                bitSystem.ListVersions();
            }
            else if (command == "List Changes") {
                bitSystem.ListChanges();
            }
            else {
                printHelp();
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
