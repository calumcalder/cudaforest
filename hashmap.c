#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hashmap.h"
#include "data.h"

#define KEY_LENGTH 20

/**
 * Function: strhash
 * -----------------
 *  Hashes a string using the djb2 algorithm (http://www.cse.yorku.ca/~oz/hash.html - last accessed 9/11/2016).
 *
 *  @param str The string to be hashed.
 *
 *  @return A numeric hash of the string.
 */
hash strhash(unsigned char* str) {
        hash h = 5381;
        int c;

        while ((c = *str++))
                h = ((h << 5) + h) + c; /* hash * 33 + c */

        return h;
}

int increment_hashmap(HashMap* hashmap_p, const char* key) {
        hash h = strhash((unsigned char*) key);
        h %= hashmap_p->size;

        if (strcmp(hashmap_p->entries[h].key, key) == 0)
                return hashmap_p->entries[h].val++;
        int h2 = h;
        int k = 1;
        int s = hashmap_p->size;
        for (int j = 0; j < hashmap_p->size*2; j++) {
                h2 = h + j*2 + k;
                k *= 2;
                h2 %= hashmap_p->size;
                if (strcmp(hashmap_p->entries[h2].key, key) == 0) {
                        hashmap_p->entries[h2].val++;
                        return 1;
                }
        }

        return put_hashmap(hashmap_p, key, 1);
}

int get_hashmap(HashMap* hashmap_p, const char* key) {
        hash h = strhash((unsigned char*) key) % hashmap_p->size;

        if (strcmp(hashmap_p->entries[h].key, key) == 0)
                return hashmap_p->entries[h].val;
        int h2 = h;
        int k = 1;
        for (int j = 0; j < hashmap_p->size*2; j++) {
                h2 = h + j*2 + k;
                k *= 2;
                h2 %= hashmap_p->size;
                if (strcmp(hashmap_p->entries[h2].key, key) == 0)
                        return hashmap_p->entries[h2].val;
        }

        return -1;
}

int get_sum_hashmap(HashMap* hashmap) {
        int count = 0;
        for (int i = 0; i < hashmap->size; i++)
                if (strcmp(hashmap->entries[i].key, HASHMAP_UNSET_CONST) != 0)
                        count += hashmap->entries[i].val;
        return count;
}

char* get_max_hashmap(HashMap* hashmap_p) {
        int maxv = -1;
        char* maxk = HASHMAP_UNSET_CONST;
        
        for (int i = 0; i < hashmap_p->size; i++) {
                if (hashmap_p->entries[i].val > maxv && strcmp(hashmap_p->entries[i].key, HASHMAP_UNSET_CONST) != 0) {
                        maxv = hashmap_p->entries[i].val;
                        maxk = hashmap_p->entries[i].key;
                }

        }
#ifdef DEBUG_HASHMAP_MAX
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
        printf("\n Hashmap Entries:\n");
        printf("       key | val\n");
        printf("-----------|----\n");

        for (int i = 0; i < hashmap_p->size; i++)
               strcmp(hashmap_p->entries[i].key, HASHMAP_UNSET_CONST) != 0 ? printf("% 10s | % 3i\n", hashmap_p->entries[i].key, hashmap_p->entries[i].val) : printf("");
#pragma GCC diagnostic pop
#endif 

        return maxk;
}

int put_hashmap(HashMap* hashmap_p, const char* key, int val) {
        hash h = strhash((unsigned char*) key) % hashmap_p->size;
        struct entry entry = hashmap_p->entries[h];

        // Free slot upon entry
        if (strcmp(entry.key, HASHMAP_UNSET_CONST) == 0) {
                strcpy(hashmap_p->entries[h].key, key);
                hashmap_p->entries[h].val = val;
                return 1;
        } 

        // Key already at first entry
        if (strcmp(entry.key, key) == 0) {
                hashmap_p->entries[h].val = val;
                return 1;
        }

        // Quadratic Probing
        int h2;
        int k = 1;
        for (int j = 0; j < hashmap_p->size*2; j++) {
                h2 = h + j + k;
                h2 %= hashmap_p->size;
                k *= 2;
                if (strcmp(hashmap_p->entries[h2].key, HASHMAP_UNSET_CONST) == 0) {
                        strcpy(hashmap_p->entries[h2].key, key);
                        hashmap_p->entries[h2].val = val;
                        return 1;
                }
                if (strcmp(hashmap_p->entries[h2].key, key) == 0) {
                        hashmap_p->entries[h2].val = val;
                        return 1;
                }
        }
        //TODO: Rehash here
        printf("failed %d\n",(int) h);
        return 0;
        

}

HashMap* make_hashmap(size_t size) {
        HashMap* hm = (HashMap*) malloc_debug(sizeof(HashMap), "hm");
        hm->entries = (struct entry*) malloc_debug(size*sizeof(struct entry), "entries");
        for (int i = 0; i < size; i++) {
                hm->entries[i].key = malloc_debug(KEY_LENGTH*sizeof(char), "key");
                strcpy(hm->entries[i].key, HASHMAP_UNSET_CONST);
                hm->entries[i].val = 0;
        }
        hm->size = size;
        return hm;
}

int has_key_hashmap(HashMap* hashmap_p, char* key) {
        for(int i = 0; i < hashmap_p->size; i++) {
                if (strcmp(hashmap_p->entries[i].key,key))
                        return 1;
        }
        return -1;
}

#ifdef _TEST_HASHMAP_C_
void __print_hashmap(HashMap* hm) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat"
        printf("\n Hashmap Entries:\n");
        printf("       key | val\n");
        printf("-----------|----\n");

        for (int i = 0; i < hm->size; i++)
               strcmp(hm->entries[i].key, HASHMAP_UNSET_CONST) != 0 ? printf("% 10s | % 3i\n", hm->entries[i].key, hm->entries[i].val) : printf("");
#pragma GCC diagnostic pop
}

int main(int argc, char* argv[]) {
        HashMap* hm = make_hashmap(32);
        put_hashmap(hm, "test1", 2);
        put_hashmap(hm, "test1", 4);
        put_hashmap(hm, "test2", 8);
        put_hashmap(hm, "test3", -16);
        put_hashmap(hm, "test4", -32);

        printf("test2: %i\n", get_hashmap(hm, "test2"));
        printf("test3: %i\n", get_hashmap(hm, "test3"));
        printf("test3: %i\n", get_hashmap(hm, "test3"));
        printf("test4: %i\n", get_hashmap(hm, "test4"));
        printf("test2: %i\n", get_hashmap(hm, "test2"));

        printf("incrementing Test2\n");
        increment_hashmap(hm, "test2");
        printf("test2: %i\n", get_hashmap(hm, "test2"));

        __print_hashmap(hm);

        HashMap* hm2 = make_hashmap(64);
        __print_hashmap(hm);
        __print_hashmap(hm2);
        put_hashmap(hm2, "test1", 5);
        put_hashmap(hm2, "test1", 6);
        put_hashmap(hm2, "test2", 7);
        put_hashmap(hm2, "test3", 8);
        put_hashmap(hm2, "test4", 9);
        increment_hashmap(hm2, "test4");

        __print_hashmap(hm);
        __print_hashmap(hm2);

        for (int i = 0; i < 1064; i++) {
                HashMap* hm3 = make_hashmap(64);
        }
}
#endif
