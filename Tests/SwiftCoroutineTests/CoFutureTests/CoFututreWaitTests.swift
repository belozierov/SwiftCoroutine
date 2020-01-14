//
//  CoFututreWaitTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 13.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFututreWaitTests: XCTestCase {
    
    func testWait() {
        testWait(sec: 1) { promise in
            XCTAssertEqual(try? promise.wait(), 1)
        }
    }
    
    func testWaitTimeout() {
        testWait(sec: 1) { promise in
            XCTAssertEqual(try? promise.wait(timeout: .now() + 2), 1)
        }
    }
    
    func testWaitTimeout2() {
        testWait(sec: 2) { promise in
            do {
                _ = try promise.wait(timeout: .now() + 1)
            } catch let error as CoFutureError {
                return XCTAssertEqual(error, .timeout)
            }
            XCTFail()
        }
    }
    
    private func testWait(sec: Int, test: @escaping (CoPromise<Int>) throws -> Void) {
        let promise = CoPromise<Int>()
        DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(sec)) {
            promise.send(1)
        }
        do {
            try test(promise)
        } catch {
            XCTFail()
        }
    }
    
}
