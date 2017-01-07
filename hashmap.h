#ifndef HASHMAP_H_
#define HASHMAP_H_
#define HASHMAP_UNSET_CONST "__NONE__"
/**
 * Type: HashMap
 * -------------
 * TODO: HashMap type docs
 */
typedef struct hashmap_s {
        struct entry {
                char* key;
                int val;
        }* entries;
        size_t size;
} HashMap;

typedef unsigned long hash;

int increment_hashmap(HashMap* hashmap_p, const char* key);
int get_hashmap(HashMap* hashmap_p, const char* key);
int put_hashmap(HashMap* hashmap_p, const char* key, int val);
char* get_max_hashmap(HashMap* hashmap_p);
int get_sum_hashmap(HashMap* hashmap_p);
int has_key_hashmap(HashMap* hashmap_p, char* key);
HashMap* make_hashmap(size_t size);

#endif
