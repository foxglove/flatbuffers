//! Builds reflection schemas for tests.

use crate::reflection_generated::reflection::{
    BaseType, Enum, EnumArgs, EnumVal, EnumValArgs, Field, FieldArgs, Object, ObjectArgs, Schema,
    SchemaArgs, Type, TypeArgs,
};
use flatbuffers::{FlatBufferBuilder, WIPOffset};

pub struct TestField {
    name: &'static str,
    base_type: BaseType,
    index: i32,
    offset: u16,
}

pub fn field(name: &'static str, base_type: BaseType, index: i32, offset: u16) -> TestField {
    TestField {
        name,
        base_type,
        index,
        offset,
    }
}

pub struct TestObject {
    name: &'static str,
    is_struct: bool,
    bytesize: i32,
    fields: Vec<TestField>,
}

impl TestObject {
    pub fn new_table(name: &'static str, fields: Vec<TestField>) -> Self {
        Self {
            name,
            is_struct: false,
            bytesize: 0,
            fields,
        }
    }

    pub fn new_struct(name: &'static str, bytesize: i32, fields: Vec<TestField>) -> Self {
        Self {
            name,
            is_struct: true,
            bytesize,
            fields,
        }
    }
}

/// Builds a schema with `objects`, which must be sorted by name, with fields sorted by name. The
/// object named `Root`, if any, is the root table. If `string_union` is set, the schema has a union
/// enum `U` whose only member is a string.
pub fn build_schema(objects: &[TestObject], string_union: bool) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let mut object_offsets = Vec::new();
    let mut root_table = None;
    for object in objects {
        let mut field_offsets = Vec::new();
        for f in &object.fields {
            let field_name = builder.create_string(f.name);
            let type_ = Type::create(
                &mut builder,
                &TypeArgs {
                    base_type: f.base_type,
                    index: f.index,
                    ..Default::default()
                },
            );
            field_offsets.push(Field::create(
                &mut builder,
                &FieldArgs {
                    name: Some(field_name),
                    type_: Some(type_),
                    offset: f.offset,
                    ..Default::default()
                },
            ));
        }
        let name = builder.create_string(object.name);
        let fields = builder.create_vector(&field_offsets);
        let offset = Object::create(
            &mut builder,
            &ObjectArgs {
                name: Some(name),
                fields: Some(fields),
                is_struct: object.is_struct,
                minalign: 1,
                bytesize: object.bytesize,
                ..Default::default()
            },
        );
        if object.name == "Root" {
            root_table = Some(offset);
        }
        object_offsets.push(offset);
    }
    let mut enum_offsets = Vec::new();
    if string_union {
        let mut values = Vec::new();
        for (name, base_type, value) in [("NONE", BaseType::None, 0), ("Str", BaseType::String, 1)]
        {
            let name = builder.create_string(name);
            let union_type = Type::create(
                &mut builder,
                &TypeArgs {
                    base_type,
                    ..Default::default()
                },
            );
            values.push(EnumVal::create(
                &mut builder,
                &EnumValArgs {
                    name: Some(name),
                    value,
                    union_type: Some(union_type),
                    ..Default::default()
                },
            ));
        }
        let name = builder.create_string("U");
        let values = builder.create_vector(&values);
        let underlying_type = Type::create(
            &mut builder,
            &TypeArgs {
                base_type: BaseType::UType,
                ..Default::default()
            },
        );
        enum_offsets.push(Enum::create(
            &mut builder,
            &EnumArgs {
                name: Some(name),
                values: Some(values),
                is_union: true,
                underlying_type: Some(underlying_type),
                ..Default::default()
            },
        ));
    }
    let objects = builder.create_vector(&object_offsets);
    let enums = builder.create_vector::<WIPOffset<Enum>>(&enum_offsets);
    let schema = Schema::create(
        &mut builder,
        &SchemaArgs {
            objects: Some(objects),
            enums: Some(enums),
            root_table,
            ..Default::default()
        },
    );
    builder.finish_minimal(schema);
    builder.finished_data().to_vec()
}
