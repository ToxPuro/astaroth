/*
    Particle sets: a source pass run on the preprocessed DSL before parsing.

    Declaration
    -----------
        Particles fp[AC_npar] migrated(par)
        {
            evolved real xp, yp, zp
            real ipar
            packonly real icell
        }
        Particles par[AC_npar]
        {
            real buf[PARTICLES_fp_NREC], idx, idx2
            output real leaving
        }

    is lowered to one gmem array per member (structure of arrays), AC_<set>_<member>, with the leading
    dimensions of a member before the particle index: real buf[n] -> gmem real AC_par_buf[n][AC_npar].
    Qualifiers of members:
      evolved   real member with the accumulator of the low-storage RK scheme AC_d<set>_<member>
      output    read by the host: kept allocated even if no kernel reads it (see the keep-alive kernel below)
      packonly  in the migration records, but not moved or unpacked

    Migration (migrated(<via>))
    ---------------------------
    The records of a particle hold the members in declaration order, followed by the accumulators of the
    evolved members. The pass declares const int PARTICLES_<set>_NREC and PARTICLES_<set>_REC_<entry>
    (entry: member, or d<member> for an accumulator) and generates
      Kernel particles_pack()             records of the particles AC_<via>_idx[i] -> AC_<via>_buf[:][i]
      Kernel particles_move()             particles AC_<via>_idx[i] -> AC_<via>_idx2[i]
      Kernel particles_unpack(int offset) records AC_<via>_buf[:][i] -> particle offset+i
    so <via> must have the members buf[PARTICLES_<set>_NREC], idx and idx2.

    Kernels over a set and member access
    ------------------------------------
        Kernel name(params) over fp { ... }
    is launched over the particles; k = vertexIdx.x is the particle index in the body.
    <set>.<member> is AC_<set>_<member>[...][k] in a kernel over a set of the same capacity, and
    <set>.<member>[...][i] is AC_<set>_<member>[...][i] everywhere. d<set>.<member> is the accumulator.

    Host header
    -----------
    user_particle_sets.h defines the record constants and PARTICLES_<set>_ARRAYS, the list of all gmem arrays of a set.

    The pass runs after the C preprocessor, so members can be conditional with #if. The lowered code keeps
    the line structure: generated declarations stay on the line of their member and the generated functions
    and kernels go on the line closing the last set, so line numbers in later error messages stay correct.
*/
#include "particle_sets.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ParticleSet particle_sets[MAX_PARTICLE_SETS];
size_t num_particle_sets = 0;

static const char* full_source = NULL; //for the line numbers of errors in the member accesses

static void
particles_error(const size_t line_num, const char* format, ...)
{
	va_list args;
	va_start(args, format);
	fprintf(stderr, "acc: error in particle sets (line %zu of the preprocessed source): ", line_num);
	vfprintf(stderr, format, args);
	fprintf(stderr, "\n");
	va_end(args);
	exit(EXIT_FAILURE);
}

static size_t
line_of(const char* pos)
{
	size_t line = 1;
	for (const char* p = full_source; p && p < pos; ++p) line += (*p == '\n');
	return line;
}

static char*
skip_space(char* s)
{
	while (*s && isspace((unsigned char)*s)) ++s;
	return s;
}

static void
trim_end(char* s)
{
	size_t len = strlen(s);
	while (len > 0 && isspace((unsigned char)s[len-1])) s[--len] = '\0';
}

static bool
is_identifier_char(const char c)
{
	return isalnum((unsigned char)c) || c == '_';
}

static bool
is_identifier_start(const char c)
{
	return isalpha((unsigned char)c) || c == '_';
}

//Copies an identifier from *s to dst and advances *s past it
static bool
read_identifier(char** s, char* dst, const size_t dst_len)
{
	char* p = skip_space(*s);
	size_t len = 0;
	if (!is_identifier_start(*p)) return false;
	while (is_identifier_char(p[len])) ++len;
	if (len >= dst_len) return false;
	memcpy(dst, p, len);
	dst[len] = '\0';
	*s = p + len;
	return true;
}

//Index of the bracket or parenthesis closing the one at s[open]
static size_t
matching_close(const char* s, const size_t open, const size_t end)
{
	const char o = s[open];
	const char c = (o == '[') ? ']' : (o == '(') ? ')' : '}';
	int depth = 0;
	for (size_t i = open; i < end; ++i) {
		if (s[i] == o) ++depth;
		else if (s[i] == c && --depth == 0) return i;
	}
	particles_error(line_of(s + open), "unbalanced %c", o);
	return end;
}

//Copies the bracketed dimensions starting at *s, e.g. "[6][3]", to dst, advances *s past them and returns their number
static int
read_dims(char** s, char* dst, const size_t dst_len, const size_t line_num)
{
	char* p = skip_space(*s);
	size_t len = 0;
	int rank = 0;
	while (*p == '[') {
		int depth = 0;
		char* q = p;
		do {
			if (*q == '[') ++depth;
			else if (*q == ']') --depth;
			else if (*q == '\0') particles_error(line_num, "unbalanced [ in dimensions");
			++q;
		} while (depth > 0);
		const size_t n = q - p;
		if (len + n >= dst_len) particles_error(line_num, "dimensions too long");
		memcpy(dst + len, p, n);
		len += n;
		++rank;
		p = skip_space(q);
	}
	dst[len] = '\0';
	*s = p;
	return rank;
}

static ParticleSet*
find_set(const char* name)
{
	for (size_t i = 0; i < num_particle_sets; ++i)
		if (!strcmp(particle_sets[i].name, name)) return &particle_sets[i];
	return NULL;
}

static ParticleMember*
find_member(ParticleSet* set, const char* name)
{
	for (size_t i = 0; i < set->num_members; ++i)
		if (!strcmp(set->members[i].name, name)) return &set->members[i];
	return NULL;
}

static bool
starts_with_word(const char* s, const char* word)
{
	const size_t len = strlen(word);
	return !strncmp(s, word, len) && !is_identifier_char(s[len]);
}

/***************************************************************************************************
  Declarations
***************************************************************************************************/

//"Particles name[capacity] [migrated(via)]" with the opening brace possibly on the same line
static void
parse_header(char* line, ParticleSet* set, bool* brace_seen, const size_t line_num)
{
	char* p = skip_space(line) + strlen("Particles");
	if (!read_identifier(&p, set->name, sizeof(set->name)))
		particles_error(line_num, "expected the name of the particle set after Particles");
	p = skip_space(p);
	if (*p != '[') particles_error(line_num, "expected [capacity] after Particles %s", set->name);
	char dims[MAX_PARTICLE_STR];
	if (read_dims(&p, dims, sizeof(dims), line_num) != 1)
		particles_error(line_num, "expected exactly one [capacity] after Particles %s", set->name);
	dims[strlen(dims)-1] = '\0'; //drop the brackets
	snprintf(set->capacity, sizeof(set->capacity), "%s", dims + 1);
	p = skip_space(p);
	if (starts_with_word(p, "migrated")) {
		p = skip_space(p + strlen("migrated"));
		if (*p != '(') particles_error(line_num, "expected migrated(<set with the migration buffers>)");
		++p;
		if (!read_identifier(&p, set->migrated_via, sizeof(set->migrated_via)))
			particles_error(line_num, "expected migrated(<set with the migration buffers>)");
		p = skip_space(p);
		if (*p != ')') particles_error(line_num, "expected ) after migrated(%s", set->migrated_via);
		p = skip_space(p + 1);
	}
	*brace_seen = false;
	if (*p == '{') {
		*brace_seen = true;
		p = skip_space(p + 1);
	}
	if (*p != '\0') particles_error(line_num, "unexpected '%s' after Particles %s[%s]", p, set->name, set->capacity);
}

static void
print_member_arrays(FILE* out, const ParticleSet* set, const ParticleMember* m)
{
	fprintf(out, "gmem %s AC_%s_%s%s[%s] ", m->type, set->name, m->name, m->dims, set->capacity);
	if (m->evolved)
		fprintf(out, "gmem %s AC_d%s_%s%s[%s] ", m->type, set->name, m->name, m->dims, set->capacity);
}

//"[evolved|output|packonly]* type name[dims], name, ..." -> gmem declarations written to out
static void
parse_members(char* line, ParticleSet* set, FILE* out, const size_t line_num)
{
	char* p = line;
	char word[MAX_PARTICLE_STR];
	bool evolved = false, output = false, packonly = false;
	while (true) {
		if (!read_identifier(&p, word, sizeof(word)))
			particles_error(line_num, "expected a member declaration in Particles %s, got '%s'", set->name, line);
		if (!strcmp(word, "evolved")) evolved = true;
		else if (!strcmp(word, "output")) output = true;
		else if (!strcmp(word, "packonly")) packonly = true;
		else break;
	}
	char type[MAX_PARTICLE_STR];
	snprintf(type, sizeof(type), "%s", word);
	if (evolved && strcmp(type, "real"))
		particles_error(line_num, "evolved members must be real in Particles %s", set->name);
	if (evolved && packonly)
		particles_error(line_num, "a member cannot be both evolved and packonly in Particles %s", set->name);

	while (true) {
		if (set->num_members == MAX_PARTICLE_MEMBERS)
			particles_error(line_num, "more than %d members in Particles %s", MAX_PARTICLE_MEMBERS, set->name);
		ParticleMember* m = &set->members[set->num_members];
		if (!read_identifier(&p, m->name, sizeof(m->name)))
			particles_error(line_num, "expected a member name in Particles %s", set->name);
		m->rank = read_dims(&p, m->dims, sizeof(m->dims), line_num);
		if (find_member(set, m->name))
			particles_error(line_num, "member %s declared twice in Particles %s", m->name, set->name);
		if (set->migrated_via[0] && m->rank > 0)
			particles_error(line_num, "members of the migrated set %s cannot have dimensions (%s%s)", set->name, m->name, m->dims);
		if (packonly && !set->migrated_via[0])
			particles_error(line_num, "packonly member %s in Particles %s, which is not migrated", m->name, set->name);
		snprintf(m->type, sizeof(m->type), "%s", type);
		m->evolved = evolved;
		m->output = output;
		m->packonly = packonly;
		++set->num_members;
		print_member_arrays(out, set, m);

		p = skip_space(p);
		if (*p == '\0') break;
		if (*p != ',') particles_error(line_num, "expected , or the end of the line after %s, got '%s'", m->name, p);
		++p;
	}
}

//Record entries of a migrated set: the members, then the accumulators of the evolved members
typedef struct {
	const ParticleMember* member;
	bool accumulator;
} RecordEntry;

static size_t
record_entries(const ParticleSet* set, RecordEntry* entries)
{
	size_t n = 0;
	for (size_t i = 0; i < set->num_members; ++i) entries[n++] = (RecordEntry){&set->members[i], false};
	for (size_t i = 0; i < set->num_members; ++i)
		if (set->members[i].evolved) entries[n++] = (RecordEntry){&set->members[i], true};
	return n;
}

static void
print_record_constants(FILE* out, const ParticleSet* set)
{
	RecordEntry entries[2*MAX_PARTICLE_MEMBERS];
	const size_t n = record_entries(set, entries);
	fprintf(out, "const int PARTICLES_%s_NREC = %zu ", set->name, n);
	for (size_t i = 0; i < n; ++i)
		fprintf(out, "const int PARTICLES_%s_REC_%s%s = %zu ", set->name, entries[i].accumulator ? "d" : "",
		        entries[i].member->name, i);
}

//Returns the lowered source, *insert_line is the line (0-based) closing the last set or -1
static char*
lower_declarations(FILE* in, long* insert_line)
{
	char* result = NULL;
	size_t result_size = 0;
	FILE* out = open_memstream(&result, &result_size);

	enum {OUTSIDE, EXPECT_BRACE, INSIDE} state = OUTSIDE;
	ParticleSet* set = NULL;
	char* buf = NULL;
	size_t buf_len = 0;
	size_t line_num = 0;
	*insert_line = -1;
	while (getline(&buf, &buf_len, in) != -1) {
		++line_num;
		trim_end(buf); //buf loses its newline here (before skip_space, which would skip it on an empty line)
		char* line = skip_space(buf);
		//Line markers of the preprocessor are kept as they are, also inside a set
		if (*line == '#') {
			fprintf(out, "%s\n", buf);
			continue;
		}
		if (state == OUTSIDE) {
			if (!starts_with_word(line, "Particles")) {
				fprintf(out, "%s\n", buf);
				continue;
			}
			if (num_particle_sets == MAX_PARTICLE_SETS)
				particles_error(line_num, "more than %d particle sets", MAX_PARTICLE_SETS);
			set = &particle_sets[num_particle_sets];
			memset(set, 0, sizeof(*set));
			bool brace_seen = false;
			parse_header(line, set, &brace_seen, line_num);
			if (find_set(set->name))
				particles_error(line_num, "particle set %s declared twice", set->name);
			state = brace_seen ? INSIDE : EXPECT_BRACE;
			fprintf(out, "\n");
		}
		else if (state == EXPECT_BRACE) {
			if (*line != '\0' && strcmp(line, "{"))
				particles_error(line_num, "expected { after Particles %s[%s]", set->name, set->capacity);
			if (*line != '\0') state = INSIDE;
			fprintf(out, "\n");
		}
		else {
			if (*line == '\0') { fprintf(out, "\n"); continue; }
			if (!strcmp(line, "}")) {
				if (set->num_members == 0) particles_error(line_num, "Particles %s has no members", set->name);
				//The record constants are needed by the dimensions of the migration buffer in a later set
				if (set->migrated_via[0]) print_record_constants(out, set);
				++num_particle_sets;
				state = OUTSIDE;
				*insert_line = line_num - 1;
				fprintf(out, "\n");
				continue;
			}
			parse_members(line, set, out, line_num);
			fprintf(out, "\n");
		}
	}
	if (state != OUTSIDE) particles_error(line_num, "missing } at the end of Particles %s", set->name);
	free(buf);
	fclose(out);
	return result;
}

/***************************************************************************************************
  Kernels over sets and member accesses
***************************************************************************************************/

typedef struct {
	int depth;
	ParticleSet* over;   //set of the kernel being scanned, NULL outside of kernels over a set
	int over_depth;      //brace depth outside of that kernel
} ScanState;

static void rewrite_range(const char* s, size_t begin, const size_t end, FILE* out, ScanState* st);

static void
rewrite_to(const char* s, const size_t begin, const size_t end, FILE* out, const ScanState* st)
{
	ScanState nested = *st;
	rewrite_range(s, begin, end, out, &nested);
}

static size_t
skip_ws(const char* s, size_t i, const size_t end)
{
	while (i < end && isspace((unsigned char)s[i])) ++i;
	return i;
}

static size_t
identifier_end(const char* s, size_t i, const size_t end)
{
	while (i < end && is_identifier_char(s[i])) ++i;
	return i;
}

static bool
equals(const char* s, const size_t b, const size_t e, const char* word)
{
	return strlen(word) == e - b && !strncmp(s + b, word, e - b);
}

//Kernel name(params) over set {   ->   Kernel name(params) { const int k = vertexIdx.x
//Returns the position after the opening brace or 0 if the kernel is not over a set
static size_t
try_kernel_over(const char* s, const size_t after_kernel, const size_t end, FILE* out, ScanState* st)
{
	size_t i = skip_ws(s, after_kernel, end);
	const size_t name_end = identifier_end(s, i, end);
	if (name_end == i) return 0;
	i = skip_ws(s, name_end, end);
	if (i >= end || s[i] != '(') return 0;
	const size_t close = matching_close(s, i, end);
	const size_t over_begin = skip_ws(s, close + 1, end);
	const size_t over_end = identifier_end(s, over_begin, end);
	if (!equals(s, over_begin, over_end, "over")) return 0;
	const size_t set_begin = skip_ws(s, over_end, end);
	const size_t set_end = identifier_end(s, set_begin, end);
	char name[MAX_PARTICLE_STR];
	snprintf(name, sizeof(name), "%.*s", (int)(set_end - set_begin), s + set_begin);
	ParticleSet* set = find_set(name);
	if (!set) particles_error(line_of(s + set_begin), "Kernel over the unknown particle set '%s'", name);
	const size_t brace = skip_ws(s, set_end, end);
	if (brace >= end || s[brace] != '{') particles_error(line_of(s + brace), "expected { after over %s", name);
	if (st->over) particles_error(line_of(s + after_kernel), "Kernel inside of a kernel over %s", st->over->name);

	fwrite(s + after_kernel, 1, close + 1 - after_kernel, out);
	for (size_t j = close + 1; j < brace; ++j) fputc(s[j] == '\n' ? '\n' : ' ', out); //keeps the lines
	fprintf(out, "{ const int k = vertexIdx.x ");
	st->over = set;
	st->over_depth = st->depth;
	st->depth++;
	return brace + 1;
}

//set.member[...] or dset.member[...] -> AC_<set>_<member>[...][k] or AC_d<set>_<member>[...]
//Returns the position after the access or 0 if this is not an access to a particle set
static size_t
try_member_access(const char* s, const size_t b, const size_t e, const size_t end, FILE* out, const ScanState* st)
{
	if (e >= end || s[e] != '.' || !is_identifier_start(s[e+1])) return 0;
	char name[MAX_PARTICLE_STR];
	snprintf(name, sizeof(name), "%.*s", (int)(e - b), s + b);
	ParticleSet* set = find_set(name);
	bool accumulator = false;
	if (!set && name[0] == 'd' && (set = find_set(name + 1))) accumulator = true;
	if (!set) return 0;
	const size_t mem_end = identifier_end(s, e + 1, end);
	char mname[MAX_PARTICLE_STR];
	snprintf(mname, sizeof(mname), "%.*s", (int)(mem_end - e - 1), s + e + 1);
	const ParticleMember* m = find_member(set, mname);
	if (!m) particles_error(line_of(s + b), "%s is not a member of the particle set %s", mname, set->name);
	if (accumulator && !m->evolved) particles_error(line_of(s + b), "%s.%s: %s is not evolved", name, mname, mname);

	fprintf(out, "AC_%s%s_%s", accumulator ? "d" : "", set->name, mname);
	size_t i = mem_end;
	int groups = 0;
	while (i < end && s[i] == '[') {
		const size_t close = matching_close(s, i, end);
		fputc('[', out);
		rewrite_to(s, i + 1, close, out, st);
		fputc(']', out);
		++groups;
		i = close + 1;
	}
	if (groups == m->rank) {
		const bool same_index = st->over && (st->over == set || !strcmp(st->over->capacity, set->capacity));
		if (!same_index)
			particles_error(line_of(s + b), "%s.%s needs a particle index here (only kernels over a set of capacity %s index it implicitly)",
			                name, mname, set->capacity);
		fprintf(out, "[k]");
	}
	else if (groups != m->rank + 1)
		particles_error(line_of(s + b), "%s.%s has %d dimensions and the particle index, but %d indices are given",
		                name, mname, m->rank, groups);
	return i;
}

static void
rewrite_range(const char* s, size_t i, const size_t end, FILE* out, ScanState* st)
{
	while (i < end) {
		const char c = s[i];
		if (c == '"') { //string literal
			size_t j = i + 1;
			while (j < end && s[j] != '"') j += (s[j] == '\\') ? 2 : 1;
			fwrite(s + i, 1, j + 1 - i, out);
			i = j + 1;
			continue;
		}
		if (c == '#') { //line markers of the preprocessor
			while (i < end && s[i] != '\n') fputc(s[i++], out);
			continue;
		}
		if (is_identifier_start(c) && (i == 0 || !is_identifier_char(s[i-1]))) {
			const size_t e = identifier_end(s, i, end);
			size_t prev = i;
			while (prev > 0 && (s[prev-1] == ' ' || s[prev-1] == '\t')) --prev;
			const bool after_dot = prev > 0 && s[prev-1] == '.';
			size_t next = 0;
			if (!after_dot) {
				if (equals(s, i, e, "Kernel")) {
					fwrite(s + i, 1, e - i, out);
					next = try_kernel_over(s, e, end, out, st);
					i = next ? next : e;
					continue;
				}
				next = try_member_access(s, i, e, end, out, st);
			}
			if (next) { i = next; continue; }
			fwrite(s + i, 1, e - i, out);
			i = e;
			continue;
		}
		if (c == '{') st->depth++;
		if (c == '}') {
			st->depth--;
			if (st->over && st->depth == st->over_depth) st->over = NULL;
		}
		fputc(c, out);
		++i;
	}
}

/***************************************************************************************************
  Generated functions and kernels
***************************************************************************************************/

static void
print_entry_array(FILE* out, const ParticleSet* set, const RecordEntry* e)
{
	fprintf(out, "AC_%s%s_%s", e->accumulator ? "d" : "", set->name, e->member->name);
}

//int(index) clamped to [0, capacity-1]
static void
print_clamped_index(FILE* out, const char* var, const char* expr, const char* capacity)
{
	fprintf(out, "int %s = int(%s) if (%s < 0) %s = 0 if (%s > (%s)-1) %s = (%s)-1 ", var, expr, var, var, var, capacity, var, capacity);
}

static void
print_migration(FILE* out, const ParticleSet* set)
{
	ParticleSet* via = find_set(set->migrated_via);
	if (!via) particles_error(0, "Particles %s migrated(%s): no particle set %s", set->name, set->migrated_via, set->migrated_via);
	const ParticleMember* buf = find_member(via, "buf");
	const ParticleMember* idx = find_member(via, "idx");
	const ParticleMember* idx2 = find_member(via, "idx2");
	if (!buf || buf->rank != 1 || !idx || idx->rank != 0 || !idx2 || idx2->rank != 0)
		particles_error(0, "Particles %s migrated(%s): %s needs the members buf[PARTICLES_%s_NREC], idx and idx2",
		                set->name, via->name, via->name, set->name);
	RecordEntry entries[2*MAX_PARTICLE_MEMBERS];
	const size_t n = record_entries(set, entries);
	char idx_expr[2*MAX_PARTICLE_STR];

	fprintf(out, "Kernel particles_pack() { const int i = vertexIdx.x ");
	snprintf(idx_expr, sizeof(idx_expr), "AC_%s_idx[i]", via->name);
	print_clamped_index(out, "k", idx_expr, set->capacity);
	for (size_t e = 0; e < n; ++e) {
		fprintf(out, "AC_%s_buf[%zu][i] = ", via->name, e);
		print_entry_array(out, set, &entries[e]);
		fprintf(out, "[k] ");
	}
	fprintf(out, "} ");

	fprintf(out, "Kernel particles_move() { const int i = vertexIdx.x ");
	print_clamped_index(out, "src", idx_expr, set->capacity);
	snprintf(idx_expr, sizeof(idx_expr), "AC_%s_idx2[i]", via->name);
	print_clamped_index(out, "dst", idx_expr, set->capacity);
	for (size_t e = 0; e < n; ++e) {
		if (entries[e].member->packonly) continue;
		print_entry_array(out, set, &entries[e]);
		fprintf(out, "[dst] = ");
		print_entry_array(out, set, &entries[e]);
		fprintf(out, "[src] ");
	}
	fprintf(out, "} ");

	fprintf(out, "Kernel particles_unpack(int offset) { const int i = vertexIdx.x ");
	print_clamped_index(out, "k", "offset+i", set->capacity);
	for (size_t e = 0; e < n; ++e) {
		if (entries[e].member->packonly) continue;
		print_entry_array(out, set, &entries[e]);
		fprintf(out, "[k] = AC_%s_buf[%zu][i] ", via->name, e);
	}
	fprintf(out, "} ");
}

//Never launched: reads the output members so that acc keeps them allocated
static void
print_keep_alive(FILE* out)
{
	bool any = false;
	for (size_t s = 0; s < num_particle_sets; ++s)
		for (size_t i = 0; i < particle_sets[s].num_members; ++i) any |= particle_sets[s].members[i].output;
	if (!any) return;
	fprintf(out, "Kernel particles_keep_outputs_alive() { const int k = vertexIdx.x ");
	for (size_t s = 0; s < num_particle_sets; ++s) {
		const ParticleSet* set = &particle_sets[s];
		for (size_t i = 0; i < set->num_members; ++i) {
			const ParticleMember* m = &set->members[i];
			if (!m->output) continue;
			char zeros[MAX_PARTICLE_STR] = "";
			for (int r = 0; r < m->rank; ++r) strcat(zeros, "[0]");
			fprintf(out, "if (AC_%s_%s%s[k] < -1.0e300) AC_%s_%s%s[k] = 0.0 ", set->name, m->name, zeros, set->name, m->name, zeros);
		}
	}
	fprintf(out, "} ");
}

static char*
generated_code(void)
{
	char* res = NULL;
	size_t size = 0;
	FILE* out = open_memstream(&res, &size);
	size_t migrated = 0;
	for (size_t s = 0; s < num_particle_sets; ++s) {
		if (particle_sets[s].migrated_via[0]) {
			if (++migrated > 1) particles_error(0, "only one migrated particle set is supported");
			print_migration(out, &particle_sets[s]);
		}
	}
	print_keep_alive(out);
	fclose(out);
	return res;
}

static void
write_host_header(void)
{
	FILE* out = fopen("user_particle_sets.h", "w");
	if (!out) return;
	fprintf(out, "#pragma once\n//Generated by acc from the Particles declarations of the DSL, see acc/particle_sets.c\n");
	for (size_t s = 0; s < num_particle_sets; ++s) {
		const ParticleSet* set = &particle_sets[s];
		fprintf(out, "#define PARTICLES_%s_ARRAYS ", set->name);
		for (size_t i = 0; i < set->num_members; ++i) {
			const ParticleMember* m = &set->members[i];
			fprintf(out, "%sAC_%s_%s", i ? "," : "", set->name, m->name);
			if (m->evolved) fprintf(out, ",AC_d%s_%s", set->name, m->name);
		}
		fprintf(out, "\n");
		if (!set->migrated_via[0]) continue;
		RecordEntry entries[2*MAX_PARTICLE_MEMBERS];
		const size_t n = record_entries(set, entries);
		fprintf(out, "#define PARTICLES_%s_NREC (%zu)\n", set->name, n);
		for (size_t i = 0; i < n; ++i)
			fprintf(out, "#define PARTICLES_%s_REC_%s%s (%zu)\n", set->name, entries[i].accumulator ? "d" : "", entries[i].member->name, i);
	}
	fclose(out);
}

/***************************************************************************************************/

void
lower_particle_sets(const char* filename)
{
	FILE* in = fopen(filename, "r");
	if (!in) return;
	num_particle_sets = 0; //acc may process the sources more than once
	long insert_line = -1;
	char* declared = lower_declarations(in, &insert_line);
	fclose(in);
	write_host_header();
	if (num_particle_sets == 0) {
		free(declared);
		return;
	}

	full_source = declared;
	char* rewritten = NULL;
	size_t rewritten_size = 0;
	FILE* out = open_memstream(&rewritten, &rewritten_size);
	ScanState st = {0, NULL, 0};
	rewrite_range(declared, 0, strlen(declared), out, &st);
	fclose(out);
	full_source = NULL;

	char* generated = generated_code();
	FILE* f = fopen(filename, "w");
	long line = 0;
	for (const char* p = rewritten; *p; ++p) {
		if (*p == '\n' && line++ == insert_line) fputs(generated, f);
		fputc(*p, f);
	}
	fclose(f);
	free(generated);
	free(rewritten);
	free(declared);
}
