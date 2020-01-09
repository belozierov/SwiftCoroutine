//
//  RefBox.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 09.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

@propertyWrapper
final class RefBox<T> {
    
    var wrappedValue: T
    @inlinable var projectedValue: RefBox { self }
    
    @inlinable init(wrappedValue value: T) {
        wrappedValue = value
    }
    
    @inlinable init<O>() where T == O? {
        wrappedValue = nil
    }
    
}

@propertyWrapper
struct ArcRefBox<T> {
    
    private let strong: RefBox<T>?
    private weak var _weak: RefBox<T>?
    
    @inlinable init(value: T) {
        strong = RefBox(wrappedValue: value)
        _weak = strong
    }
    
    @inlinable init(weak: RefBox<T>?) {
        strong = nil
        _weak = weak
    }
    
    @inlinable init(strong: RefBox<T>?) {
        self.strong = strong
        _weak = strong
    }
    
    @inlinable var wrappedValue: T? {
        get { _weak?.wrappedValue }
        set { newValue.map { _weak?.wrappedValue = $0 } }
    }
    
    @inlinable var projectedValue: ArcRefBox { self }
    
    @inlinable var weak: ArcRefBox {
        .init(weak: _weak)
    }
    
    @inlinable var isStrong: Bool {
        strong == nil
    }
    
}
