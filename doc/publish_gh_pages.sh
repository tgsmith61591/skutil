###########################################
##This script should be run immediately####
##after a push has been made to the master#
##branch to update the gh-pages branch#####
##You will want to run this script from####
##its current location like as follows:####
##sh publish_gh_pages.sh "commit message"##
###########################################

# This script will check if the number of input arguments are 1
#   A) If arguments == 1
#       1) make clean html                  # purge old docs and create new
#       2) git stash                        # delete unwanted/uncommited changes to master
#       3) git checkout gh-pages            # switch to gh-pages
#       4) cp -R ../gh-pages/html/* ../     # copy new html to base directory of gh-pages
#       5) git add -A                       # add the new and modified files
#       6) git commit -m "$COMMIT_MESSAGE"  # commit the added changes
#       7) git push origin gh-pages         # push the changes to gh-pages
#       8) git checkout master              # return to master
#   B) If arguments != 1
#       1) print messages on proper usage


if test $# -eq 1; then

    COMMIT_MESSAGE="$1"

    echo "\n\nSee note A.1) make clean html\n\n"
    make clean html

    echo "\n\nSee note A.2) git stash\n\n"
    git stash

    echo "\n\nSee note A.3) git checkout gh-pages\n\n"
    git checkout gh-pages
    git pull origin gh-pages

    echo "\n\nSee note A.4) cp -R ../gh-pages/html/* ../\n\n"
    cp -R ../gh-pages/html/* ../

    echo "\n\nSee note A.5) git add -A\n\n"
    git add -A

    echo "\n\nSee note A.6) git commit -m '$COMMIT_MESSAGE'\n\n"
    git commit -m "$COMMIT_MESSAGE"

    echo "\n\nSee note A.7) git push origin gh-pages\n\n"
    git push origin gh-pages

    echo "\n\nSee note A.8) git checkout master\n\n"
    git checkout master

else
    echo "\n>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<"
    echo ">>>>>>>Illegal number of arguments supplied!<<<<<<<"
    echo ">>>>>>>Please pass commit message to script<<<<<<<<"
    echo "Please invoke script like in the following example:"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<\n"
    echo "sh publish_gh_pages.sh 'commit message'\n"
fi
