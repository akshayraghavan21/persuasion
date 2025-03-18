#!/usr/bin/env python


import json
import MySQLdb
import sys
import os
from tqdm import tqdm

## Create Temporary OP Comments Table as below
"""
DROP TABLE IF EXISTS temp_op_comments;
CREATE TABLE temp_op_comments AS
SELECT 
    comments_sub_pft_v1.id as comments_id,
    submissions_sub_pft_v1.id as submission_id,
    submissions_sub_pft_v1.author as submission_author,
    submissions_sub_pft_v1.title as submission_title,
    submissions_sub_pft_v1.selftext as submission_body
FROM comments_sub_pft_v1 
JOIN submissions_sub_pft_v1 
ON SUBSTRING(comments_sub_pft_v1.link_id, 4) = submissions_sub_pft_v1.id 
AND comments_sub_pft_v1.author = submissions_sub_pft_v1.author;


-- Add index for faster lookups
CREATE INDEX idx_temp_op_id ON temp_op_comments(comments_id(7));
"""


# Sample Record in the Dataset File: 
"""
{
    "submission_id":, 
    "submission_author":, 
    "submission_title":, 
    "submission_body":, 
    "comments_thread_root_comment_id":, 
    "comments_thread_leaf_comment_id":,
    "comments_thread_is_delta":,
    "comments_thread":[
        {
            "comment_id":,
            "comment_author":,
            "comment_body":,
            "comment_parent_id":,
            "comment_link_id":,
            "comment_created_utc":,
        },
    ]
}
"""

## Recursive Query
recursive_query = """
    WITH RECURSIVE comment_thread AS (
        -- Base case: Start from the given OP comment
        SELECT c.id, c.author, c.body, c.parent_id, c.link_id, c.created_utc
        FROM comments_sub_pft_v1 c
        WHERE c.id = %s

        UNION ALL

        -- Recursive case: Find replies to comments in the previous level
        SELECT c.id, c.author, c.body, c.parent_id, c.link_id, c.created_utc
        FROM comments_sub_pft_v1 c
        INNER JOIN comment_thread ct ON c.id = SUBSTRING(ct.parent_id, 4)
    )
    SELECT * FROM comment_thread ORDER BY created_utc;
"""

# Expand the path to ~/.my.cnf
config_file = os.path.expanduser("~/.my.cnf")
output_data_file = "persuasion_12_factor_comments_thread_dataset_v0.1_20250228.json"
delta_symbols = ["&#8710;", "!delta"]

conn = MySQLdb.connect(
    db="persuasion", 
    read_default_file=config_file, 
    charset="utf8mb4", 
    use_unicode=True
)
cur = conn.cursor()

op_comments_table_query = "select * from temp_op_comments;"
cur.execute(op_comments_table_query)

rows = cur.fetchall()
print("Number of OP Comments:", len(rows))
# test_id = [row[0] for row in rows if row[0] == "j2fh1rb"][0]
# print(test_id)


# Function to check if a comment contains delta acknowledgment
def check_delta_ack_comment(comment_body):
    return any(symbol in comment_body for symbol in delta_symbols)


# Delete the file if it already exists
if os.path.exists(output_data_file):
    os.remove(output_data_file)

with open(output_data_file, "a", encoding="utf-8") as json_file:
    for row in tqdm(rows, desc="Processing Comment Threads"):
        try:
            ## Check if the tables desc is the same as below, else change accordingly
            comment_id, submission_id, submission_author, submission_title, submission_body = row

            submission_author = submission_author.decode("utf-8", "ignore") if isinstance(submission_author, bytes) else submission_author
            submission_title = submission_title.decode("utf-8", "ignore") if isinstance(submission_title, bytes) else submission_title
            submission_body = submission_body.decode("utf-8", "ignore") if isinstance(submission_body, bytes) else submission_body

            # Execute the recursive query for the current comment ID
            cur.execute(recursive_query, (comment_id,))
            
            # Fetch the results of the recursive query
            comment_thread_raw = cur.fetchall()

            # Print or process the comment thread results
            comments_thread_arr = []
            for comment in comment_thread_raw:
                comment_id, comment_author, comment_body, comment_parent_id, comment_link_id, comment_created_utc = comment

                comment_author = comment_author.decode("utf-8", "ignore") if isinstance(comment_author, bytes) else comment_author
                comment_body = comment_body.decode("utf-8", "ignore") if isinstance(comment_body, bytes) else comment_body

                comment_dict = {
                    "comment_id":comment_id,
                    "comment_author":comment_author,
                    "comment_body":comment_body,
                    "comment_parent_id":comment_parent_id,
                    "comment_link_id":comment_link_id,
                    "comment_created_utc":comment_created_utc,
                }
                comments_thread_arr.append(comment_dict)
            
            if not comments_thread_arr:
                continue  # Skip empty threads
    
            is_delta_flag = check_delta_ack_comment(comment_body=comments_thread_arr[-1]['comment_body'])
            record_dict = {
                "submission_id":submission_id, 
                "submission_author":submission_author, 
                "submission_title":submission_title, 
                "submission_body":submission_body, 
                "comments_thread_root_comment_id":comments_thread_arr[0]['comment_id'], 
                "comments_thread_leaf_comment_id":comments_thread_arr[-1]['comment_id'],
                "comments_thread_is_delta":is_delta_flag,
                "comments_thread":comments_thread_arr
            }

            ## Append record_dict to a json file: output_data_file as a record so as to not store in memory
            # Append the record to the JSON file
            json.dump(record_dict, json_file, ensure_ascii=False)
            json_file.write("\n")
            json_file.flush()

        except UnicodeDecodeError as e:
            print(f"Encoding error at comment_id: {comment_id}, skipping. Error: {e}")
        except Exception as e:
            print(f"Unexpected error at comment_id: {comment_id}: {e}")

# Close the cursor and the connection to the database
cur.close()
conn.close()
print(f"Finished Writing to File:{output_data_file}")