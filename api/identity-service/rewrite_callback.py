def commit_callback(commit):
    # Replace any author/committer email equal to ryan@example.com
    try:
        if commit.author_email == b"ryan@example.com":
            commit.author_name = b"Ryan Jewik"
            commit.author_email = b"jewik@chapman.edu"
        if commit.committer_email == b"ryan@example.com":
            commit.committer_name = b"Ryan Jewik"
            commit.committer_email = b"jewik@chapman.edu"
    except Exception:
        # be resilient to unexpected commit object shapes
        pass
