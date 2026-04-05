
def pre_clean():
    with open('../dataset/zsh_history', 'r', encoding='latin-1') as f:
        history = [';'.join(x.split(';')[1:]) for x in f.read().split('\n') if len(';'.join(x.split(';')[1:]).split(' ')) > 1]

        return history

if __name__ == '__main__':
    with open('../dataset/history_pre_clean', 'w') as f:
        f.write('\n'.join(pre_clean()))

