//
//  AtomicIntRepresentable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

@propertyWrapper @usableFromInline
struct AtomicIntRepresentable<T: RawRepresentable> where T.RawValue == Int {
    
    @usableFromInline
    @AtomicInt private(set) var rawValue: Int
    
    init(wrappedValue value: T) {
        rawValue = value.rawValue
    }
    
    var wrappedValue: T {
        @inlinable get { T(rawValue: rawValue)! }
        set { rawValue = newValue.rawValue }
    }
    
    @inlinable var projectedValue: AtomicIntRepresentable {
        get { self }
        set { self = newValue }
    }
    
    mutating func update(from: T, to: T) -> Bool {
        $rawValue.update(from: from.rawValue, to: to.rawValue)
    }
    
}


