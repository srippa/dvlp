#!/bin/zsh
emulate -LR zsh # reset zsh options
# export PATH=/usr/bin:/bin:/usr/sbin:/sbin

if [[  $(arch) == arm64 ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
    echo "Detected arm architecture - setting homebrew to /opt/homebrew/bin"
else
    export PATH="/usr/local/bin:$PATH"
    echo ""Detected intel architecture - setting homebrew to /usr/local/bin"
fi

# alias intel 'arch -x86_64 /usr/local/bin/fish'

function countArguments() {
    echo "${#@}"
}

wordlist="one two three four five"
wordarray=( $wordlist )
for word in $wordarray; do
    echo "->$word<-"
done


echo "normal substitution, no quotes:"
countArguments $wordlist
# -> 1

echo "substitution with quotes"
countArguments "$wordlist"
# -> 1